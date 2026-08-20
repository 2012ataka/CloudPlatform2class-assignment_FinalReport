import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import uvicorn
from datetime import datetime, timezone

app = FastAPI()
client = genai.Client()

class VoiceAnalysisResponse(BaseModel):
    CallDate: str | None = Field(description="対応日時。音声内に言及がない場合は必ずnullにすること。")
    ProcessDate: str | None = Field(description="処理日時（システム側で上書きするためAIはnullを出力すること）")
    Organization: str | None = Field(description="所属（会社名や部署名）")
    CustomerName: str | None = Field(description="名前（顧客名）。言及がない場合はnull。")
    PhoneNumber: str | None = Field(description="電話番号")
    Address: str | None = Field(description="住所")
    ReservationDate: str | None = Field(description="予約日や希望日など")
    Summary: str | None = Field(description="要件の要約")
    OurResponse: str | None = Field(description="こちらの回答・対応内容")
    FullTranscript: str | None = Field(description="話者を区別したやりとり全文。必ず「話者: MM:SS 発言内容」の形式で記述し、1発言ごとに改行(\n)を入れること。")
    Category: str | None = Field(description="問い合わせ種別（クレーム、予約、製品への質問、注文など）")
    Sentiment: str | None = Field(description="顧客の感情（怒り、不満、普通、満足など）")
    Urgency: str | None = Field(description="重要度（高・中・低）")
    NextAction: str | None = Field(description="ネクストアクション（折り返し、資料送付など）")
    OperatorName: str | None = Field(description="対応オペレーター名")
    CallDuration: int | None = Field(description="通話時間（秒）")
    Status: str = Field(default="未対応", description="ステータス（常に'未対応'とする）")

@app.post("/api/analyze-voice")
async def analyze_voice(request: Request):
    try:
        audio_content = await request.body()
        
        if not audio_content:
            raise HTTPException(status_code=400, detail="データが空です")

        prompt = """
        提供された音声データを解析し、指定されたJSONスキーマに従ってすべての項目を抽出してください。
        情報が含まれていない項目はnullとしてください。
        
        【重要事項】
        - CallDate（対応日時）：音声の中に日時の言及がない場合は勝手に推測せず、必ずnullにしてください。
        - ProcessDate（処理日時）：ここではnullを出力してください（システム側で自動付与します）。
        - CustomerName（顧客名）：音声の中に該当する名前の言及がない場合は無理やり抽出・推測せず、nullにしてください。
        - FullTranscript（全文）：必ずタイムスタンプごとに改行(\n)し、以下の厳密なフォーマットで出力してください。
          例：
          佐藤: 00:01
          佐藤: 00:01昨日のズボンの裾上げを依頼した佐藤です。
          佐藤: 00:04追加でジャケットの袖丈もお願いしたいのですが、その場合の追加費用と受け取れる時期がいつになるか教えていただけますでしょうか？
          オペレーター: 00:10はい、かしこまりました。
        - Status：常に「未対応」としてください。
        """
        
        audio_part = types.Part.from_bytes(data=audio_content, mime_type="audio/mp4")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[audio_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VoiceAnalysisResponse,
            ),
        )
        
        # AIの出力したJSONを辞書オブジェクトとして読み込む
        response_data = json.loads(response.text)
        
        # Python側で正確なUTC時刻を取得し、JSONのProcessDateを強制的に上書きする
        # （ODCはこの形式を受け取ると、自動的にローカルの日本時間へ変換して保存する）
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