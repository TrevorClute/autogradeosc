#!/bin/bash

BASE_URL="http://localhost:5001"
SECRET="d84b39d6c0c1d80bc104d5f22d8bc5cf"

echo "=== Health Check ==="
curl -s "$BASE_URL/health"
echo -e "\n"

echo "=== Single Essay Evaluation ==="
curl -s -X POST "$BASE_URL/evaluate" \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: $SECRET" \
  -d '{
    "essay_text": "I believe I would be a strong candidate for this program because of my experience in computational science and my passion for high-performance computing. During my undergraduate studies, I developed parallel algorithms using MPI and OpenMP, which gave me hands-on experience with distributed systems. I am eager to apply these skills in a research setting and contribute meaningfully to the work being done at OSC.",
    "prompt_id": 1
  }' | python3 -m json.tool
echo ""

echo "=== Batch Essay Evaluation ==="
curl -s -X POST "$BASE_URL/evaluate/batch" \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: $SECRET" \
  -d '{
    "essays": [
      {
        "essay_id": "essay_1.pdf",
        "essay_text": "I would benefit from this program by gaining access to cutting-edge computing resources and mentorship from experienced researchers. This opportunity would help me develop the technical skills needed for my career in data science.",
        "prompt_id": 2
      },
      {
        "essay_id": "essay_2.pdf",
        "essay_text": "In my software engineering class, I worked on a team project to build a web application. I served as the backend developer, designing the database schema and API endpoints. We succeeded in delivering on time, though we had disagreements about the tech stack. I handled the frustration by scheduling a team meeting where everyone could voice their concerns.",
        "prompt_id": 3
      }
    ]
  }' | python3 -m json.tool
echo ""

echo "=== Unauthorized Request (should return 401) ==="
curl -s -X POST "$BASE_URL/evaluate" \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: wrong-key" \
  -d '{"essay_text": "test", "prompt_id": 1}' | python3 -m json.tool
echo ""
