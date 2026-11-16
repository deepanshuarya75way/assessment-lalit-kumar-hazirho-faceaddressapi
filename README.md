# hazirho-faceaddressapi

curl -X POST "http://127.0.0.1:8000/v1/face/liveness" \
  -F "files=@/home/coder/Downloads/e1.png;type=image/png" \
  -F "files=@/home/coder/Downloads/e2.png;type=image/png" \
  -F "files=@/home/coder/Downloads/e3.png;type=image/png" | python -m json.tool