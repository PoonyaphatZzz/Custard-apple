import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, Request, HTTPException

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import MessageEvent, ImageMessageContent, TextMessageContent

app = FastAPI()

CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN", "")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET", "")

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

MODEL_PATH = "custard_apple_model.onnx"
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

CLASSES = [
    "Anthracnose",
    "Blank Canker",
    "Diplodia Rot",
    "Healthy",
    "Leaf spot on Leaves",
    "Leaf spot on fruit",
    "Mealy Bug"
]

DISEASE_INFO = {
    "Anthracnose": {
        "th": "โรคแอนแทรคโนส",
        "desc": "เกิดจากเชื้อรา ทำให้เกิดจุดแผลสีน้ำตาลเข้มหรือดำ ลุกลามเป็นวงกลม",
        "treat": "ตัดแต่งส่วนที่เป็นโรคไปเผาทำลาย และฉีดพ่นสารคอปเปอร์ไฮดรอกไซด์ หรือไดฟีโนโคนาโซล"
    },
    "Blank Canker": {
        "th": "โรคแคงเกอร์ (กิ่ง/ต้นแห้งตาย)",
        "desc": "เปลือกต้นแตก แผลตกสะเก็ด มีน้ำยางซึม กิ่งเหี่ยวแห้ง",
        "treat": "ถากเปลือกบริเวณที่เป็นแผลออกแล้วทาด้วยปูนแดง หรือสารป้องกันกำจัดเชื้อรา"
    },
    "Diplodia Rot": {
        "th": "โรคผลเน่าดิพโลเดีย",
        "desc": "ผลแห้งดำ แข็งติดค้างอยู่บนต้น เนื้อผลเน่าเสีย",
        "treat": "เก็บผลที่เป็นโรคออกจากแปลงปลูก ห่อผลตั้งแต่ระยะติดผล และพ่นสารป้องกันกำจัดเชื้อรา"
    },
    "Healthy": {
        "th": "ต้นน้อยหน่าแข็งแรง (ปกติ)",
        "desc": "ไม่พบอาการผิดปกติของโรคพืช",
        "treat": "ดูแลรักษาตามปกติ ให้น้ำและปุ๋ยสม่ำเสมอเพื่อเสริมสร้างภูมิคุ้มกัน"
    },
    "Leaf spot on Leaves": {
        "th": "โรคใบจุดบนใบ",
        "desc": "เกิดจุดสีน้ำตาลหรือเทากระจายทั่วใบ ทำให้ใบร่วงก่อนกำหนด",
        "treat": "ตัดแต่งทรงพุ่มให้โปร่ง มีอากาศถ่ายเท และพ่นสารป้องกันเชื้อรากลุ่มแมนโคเซบ"
    },
    "Leaf spot on fruit": {
        "th": "โรคจุดบนผล",
        "desc": "ผิวผลเกิดจุดด่างดำขนาดเล็ก ผิวผลไม่สวยงามและอาจเน่าเสียได้ง่าย",
        "treat": "ห่อผลน้อยหน่าเพื่อป้องกันเชื้อโรค และหลีกเลี่ยงการให้น้ำโดนผลโดยตรง"
    },
    "Mealy Bug": {
        "th": "เพลี้ยแป้ง",
        "desc": "แมลงเกาะดูดกินน้ำเลี้ยง มีคราบสีขาวคล้ายผงแป้งปกคลุม",
        "treat": "ใช้น้ำฉีดล้าง หรือพ่นด้วยน้ำยาล้างจานเจือจาง/สารสกัดสะเดา หากระบาดรุนแรงใช้สารไทอะมีทอกแซม"
    }
}

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_data = np.array(image).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@app.get("/")
def home():
    return {"status": "Custard Apple LINE Bot is running!"}

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_str = body.decode("utf-8")
    
    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    reply_text = (
        "สวัสดีครับ! 🌿 ส่งรูปภาพใบหรือผลน้อยหน่าเข้ามาในแชทได้เลยครับ "
        "ระบบจะช่วยตรวจวิเคราะห์โรคพืชพร้อมแนะนำวิธีรักษาให้อัตโนมัติครับ"
    )
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    with ApiClient(configuration) as api_client:
        blob_api = MessagingApiBlob(api_client)
        image_content = blob_api.get_message_content(message_id=event.message.id)
        
    input_tensor = preprocess_image(image_content)
    outputs = session.run(None, {input_name: input_tensor})
    scores = outputs[0][0]
    
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    
    predicted_class = CLASSES[pred_idx]
    info = DISEASE_INFO.get(predicted_class, {})
    
    result_text = (
        f"🌿 ผลการวินิจฉัย: {info.get('th', predicted_class)}\n"
        f"📊 ความมั่นใจ: {confidence:.2f}%\n\n"
        f"🔍 รายละเอียด:\n{info.get('desc', '-')}\n\n"
        f"💊 คำแนะนำและการรักษา:\n{info.get('treat', '-')}"
    )
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=result_text)]
            )
        )
