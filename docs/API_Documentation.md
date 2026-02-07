## 📘 Intelligent Sri Lanka IRD Tax Intelligence & Compliance Assistant

### API Documentation

---

## 1. Overview

The **IRD Tax Intelligence API** provides a secure, document-grounded interface for querying Sri Lanka Inland Revenue Department (IRD) tax information.

_This API is part of the Intelligent Sri Lanka IRD Tax Intelligence & Compliance Assistant project._

⚠️ **Important Guarantee**
All answers are generated **strictly from IRD-published PDF documents**.
The system **does not guess**, **does not hallucinate**, and **does not use external knowledge**.

If information is unavailable or ambiguous, the API clearly informs the user.

---

## 2. Base URL

```
http://127.0.0.1:8000
```

---

## 3. Authentication

No authentication is required (local academic prototype).

---

## 4. API Endpoints

---

## 🔹 4.1 Upload IRD Documents

### Endpoint

```
POST /upload
```

### Purpose

Upload one or more **official IRD PDF documents** and automatically rebuild:

* Text extraction
* Chunking
* Vector embeddings
* FAISS index

---

### Request

**Content-Type:** `multipart/form-data`

| Field | Type   | Description               |
| ----- | ------ | ------------------------- |
| files | File[] | One or more IRD PDF files |

---

### Example (Swagger / Postman)

Upload:

```
PN_IT_2025-01.pdf
SET_25_26_Detail_Guide_E.pdf
```

---

### Response (200 OK)

```json
{
  "message": "Uploaded and rebuilt index successfully",
  "files": [
    "PN_IT_2025-01.pdf",
    "SET_25_26_Detail_Guide_E.pdf"
  ]
}
```

---

### Possible Errors

| Status | Meaning              |
| ------ | -------------------- |
| 400    | File is not a PDF    |
| 500    | Index rebuild failed |

---

## 🔹 4.2 Query Tax Information

### Endpoint

```
POST /query
```

---

### Purpose

Ask tax-related questions using IRD documents only.

The API automatically:

1. Retrieves relevant document sections
2. Applies guardrails
3. Generates a safe answer
4. Adds citations
5. Appends a mandatory disclaimer

---

### Request Body

```json
{
  "question": "What is the personal relief for 2025/2026?",
  "top_k": 5,
  "use_llm": true
}
```

| Field    | Type    | Description                     |
| -------- | ------- | ------------------------------- |
| question | string  | User tax question               |
| top_k    | number  | Number of chunks to retrieve    |
| use_llm  | boolean | Enable LLM answering (optional) |

---

### Response Structure

```json
{
  "answer": "…",
  "sources": [],
  "status": "ok",
  "used_llm": true
}
```

| Field    | Description              |
| -------- | ------------------------ |
| answer   | Final response text      |
| sources  | List of cited documents  |
| status   | ok / missing / ambiguous |
| used_llm | Whether LLM was used     |

---

## 5. Example Scenarios

---

### ✅ Example 1: Valid Question

**Request**

```json
{
  "question": "What is the personal relief for 2025/2026?"
}
```

**Response**

```json
{
  "answer": "Personal Relief for the year of assessment 2025/2026 is Rs. 1,800,000.",
  "sources": [
    {
      "document": "PN_IT_2025-01_26032025_E.pdf",
      "page_range": [1],
      "section": "PN/IT/2025-01"
    }
  ],
  "status": "ok",
  "used_llm": true
}
```

---

### ❌ Example 2: Missing Information

**Request**

```json
{
  "question": "What is the VAT rate for 2025?"
}
```

**Response**

```json
{
  "answer": "This information is not available in the provided IRD documents.\n\nThis response is based solely on IRD-published documents and is not professional tax advice.",
  "sources": [],
  "status": "missing",
  "used_llm": false
}
```

---

### ⚠️ Example 3: Ambiguous Question

**Request**

```json
{
  "question": "What is the tax rate?"
}
```

**Response**

```json
{
  "answer": "Your question may be ambiguous. Do you mean Personal Income Tax (PIT), Corporate Income Tax (CIT), or SET instructions? Please specify the tax type and year of assessment.",
  "sources": [],
  "status": "ambiguous",
  "used_llm": false
}
```

---

## 6. Guardrails & Safety

The API enforces **three critical guardrails**:

| Condition       | Behavior                |
| --------------- | ----------------------- |
| Missing info    | Returns “Not available” |
| Ambiguous query | Requests clarification  |
| Low confidence  | Refuses to answer       |

This guarantees **zero hallucination**.

---

## 7. Disclaime

Every response automatically ends with:

> **“This response is based solely on IRD-published documents and is not professional tax advice.”**

This is enforced at code level.

---

## 8. Testing the API

### Swagger UI

```
http://127.0.0.1:8000/docs
```

### CLI Testing

```bash
python -m llm.answer_with_llm
```

---

## 9. Limitations

* Only IRD PDFs are used
* No external tax knowledge
* LLM runs locally (Ollama)
* Not a replacement for a tax professional

---

## 10. Version

**API Version:** 1.0.0
**Project Type:** Academic / Research Prototype

---
