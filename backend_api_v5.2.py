import os
import json
import tempfile
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import uvicorn
from datetime import datetime, timezone
from faster_whisper import WhisperModel

app = FastAPI()
client = genai.Client()

# Whisperモデルの事前読み込み（音声解析用）
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# --- 音声解析用スキーマ ---
class VoiceAnalysisResponse(BaseModel):
    CallDate: str | None = Field(description="対応日時。言及がない場合は必ずnullにすること。")
    ProcessDate: str | None = Field(description="処理日時（システム側で上書きするためAIはnullを出力）")
    Organization: str | None = Field(description="所属（会社名や部署名）")
    CustomerName: str | None = Field(description="名前（顧客名）。言及がない場合のみnull。")
    PhoneNumber: str | None = Field(description="電話番号")
    Address: str | None = Field(description="住所")
    ReservationDate: str | None = Field(description="予約日や希望日など")
    Summary: str | None = Field(description="要件の要約")
    OurResponse: str | None = Field(description="こちらの回答・対応内容")
    FullTranscript: str | None = Field(description="話者を区別したやりとり全文。提供されたテキストを一言一句省略せず、文脈から話者を推測して「話者: 発言内容」の形式で記述すること。改行(\\n)を入れること。")
    Category: str | None = Field(description="問い合わせ種別")
    Sentiment: str | None = Field(description="顧客の感情")
    Urgency: str | None = Field(description="重要度（高・中・低）")
    NextAction: str | None = Field(description="ネクストアクション")
    OperatorName: str | None = Field(description="対応オペレーター名")
    CallDuration: int | None = Field(description="通話時間（秒）")
    Status: str = Field(default="未対応", description="ステータス（常に'未対応'）")

# --- 書類解析用スキーマ ---
class DocumentAnalysisResponse(BaseModel):
    DocumentDate: str | None = Field(description="書類内に記載されている日付（例：2026年8月15日）。記載がない場合や認識できない場合は必ずnullとすること。")
    ProcessDate: str | None = Field(description="処理日時（システム側で上書きするためAIはnullを出力）")
    DocumentType: str | None = Field(description="書類の種別（例：領収書、レシート、学校への申請書、配布マニュアルなど）")
    RelatedParty: str | None = Field(description="関係者や関係組織（例：宛先、発行元、提出先、店舗名など）。記載がない場合はnullにすること。")
    Amount: int | None = Field(description="書類に記載されている合計金額。金額の記載がない書類の場合は必ずnullにすること。")
    Summary: str | None = Field(description="書類の内容の要約（簡潔に）。")
    ParsedContent: str | None = Field(description="書類の内容を構造化したテキスト全文。書類の種別に応じて、適切な見出し（例：「品名」「申請理由」「項目」など）を付け、データとしてコピー＆ペーストしやすい形式に整形すること。")

# --- 音声解析エンドポイント ---
@app.post("/api/analyze-voice")
async def analyze_voice(request: Request):
    temp_file_path = ""
    try:
        audio_content = await request.body()
        if not audio_content:
            raise HTTPException(status_code=400, detail="データが空です")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        segments, info = whisper_model.transcribe(temp_file_path, beam_size=5)
        raw_transcript = "".join([segment.text for segment in segments])

        if not raw_transcript.strip():
            raise HTTPException(status_code=400, detail="音声からテキストを抽出できませんでした")

        prompt = """
        提供された以下の文字起こしテキストデータを解析し、指定されたJSONスキーマに従ってすべての項目を抽出してください。
        音声の生の書き起こしであるため、えー、あー等の相槌が含まれています。
        
        【文字起こしテキスト】
        """ + raw_transcript + """

        【重要事項】
        - CallDate：テキスト内に日時の言及がない場合は必ずnullにしてください。
        - ProcessDate：nullを出力してください。
        - FullTranscript：文字起こしテキストの内容を一言一句省略・要約せず、文脈から話者（例：顧客、オペレーター等）を推測して「話者: 発言内容」の形式に整形し、すべて書き起こしてください。
        - Status：常に「未対応」としてください。
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VoiceAnalysisResponse,
            ),
        )
        
        response_data = json.loads(response.text)
        response_data["ProcessDate"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
         
        return JSONResponse(
            content=response_data,
            media_type="application/json; charset=utf-8"
        )
                
    except Exception as e:
        print(f"エラー発生: {str(e)}")
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- 書類解析エンドポイント ---
@app.post("/api/analyze-document")
async def analyze_document(request: Request):
    try:
        image_content = await request.body()
        if not image_content:
            raise HTTPException(status_code=400, detail="データが空です")

        prompt = """
        提供された書類の画像を解析し、指定されたJSONスキーマに従って情報を抽出してください。
        書類の内容をデータ化し、システム上でコピー＆ペースト等を行いやすくすることが目的です。

        【重要事項】
        - DocumentDate：書類に記載されている日付や日時を抽出し、「YYYY年MM月DD日」形式に整形してください。記載がない場合、読み取れない場合、または不明な場合は必ずnullにしてください（空欄にするため）。
        - ProcessDate：nullを出力してください。
        - DocumentType：書類の種類（領収書、申請書など）を判定して記載してください。
        - RelatedParty：書類に関わる宛名、店舗名、提出先などを抽出してください。
        - Amount：書類に金額の記載がある場合のみ数値を抽出し、記載がない場合はnullにしてください。
        - Summary：書類の要点や目的を簡潔に要約してください。
        - ParsedContent：書類の内容を構造化し、書類の種類に応じた見出しを付けてテキストとして整理してください。
        """

        image_part = types.Part.from_bytes(data=image_content, mime_type="image/jpeg")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DocumentAnalysisResponse,
            ),
        )
        
        response_data = json.loads(response.text)
        
        # DocumentDateがNoneまたは空欄の場合は確実に空文字列に変換
        if not response_data.get("DocumentDate"):
            response_data["DocumentDate"] = ""

        response_data["ProcessDate"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return JSONResponse(
            content=response_data,
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        print(f"エラー発生: {str(e)}")
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)