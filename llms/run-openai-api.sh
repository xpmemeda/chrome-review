curl http://29.39.224.137:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tfcc-deepseek-r1" \
  -d '{
    "model": "DeepSeek-R1",
    "messages": [{"role": "user", "content": "你好！"}],
    "max_tokens": 8,
    "temperature": 1.0
  }'