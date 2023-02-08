from typing import Dict, List, Any
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64

from facenet_pytorch import MTCNN, InceptionResnetV1

def init():
    global device
    global mtcnn
    global resnet

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

def inference(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    global device
    global mtcnn
    global resnet

    imageData = data.get("inputs").get("image")
    image = Image.open(BytesIO(base64.b64decode(imageData)))
    face_batch = mtcnn([image])
    face_batch = [i for i in face_batch if i is not None]
    if face_batch:
        aligned = torch.stack(face_batch)
        if device.type == "cuda":
            aligned = aligned.to(device)

        embeddings = resnet(aligned).detach().cpu()
        return embeddings.tolist()
    else: return None
