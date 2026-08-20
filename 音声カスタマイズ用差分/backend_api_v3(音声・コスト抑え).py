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

# Whisperモデルの事前読み込み（CPU環境を想定し、高速かつ軽量な'small'モデルを指定）
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

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

@app.post("/api/analyze-voice")
async def analyze_voice(request: Request):
    temp_file_path = ""
    try:
        audio_content = await request.body()
        if not audio_content:
            raise HTTPException(status_code=400, detail="データが空です")

        # 1. 音声データを一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        # 2. faster-whisperでテキスト化（音声認識）
        segments, info = whisper_model.transcribe(temp_file_path, beam_size=5)
        raw_transcript = "".join([segment.text for segment in segments])

        if not raw_transcript.strip():
            raise HTTPException(status_code=400, detail="音声からテキストを抽出できませんでした")

        # 3. Geminiへテキストを渡して情報抽出および話者整形
        prompt = f"""
        提供された以下の文字起こしテキストデータを解析し、指定されたJSONスキーマに従ってすべての項目を抽出してください。
        音声の生の書き起こしであるため、えー、あー等の相槌が含まれています。
        
        【文字起こしテキスト】
        {raw_transcript}

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
        # 一時ファイルのクリーンアップ
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)