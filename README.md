# Red Flags API

## Get started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Insall pre-commit hooks

```bash
pre-commit install
```

### # Add anthropic API key

1. Create `.env` file in the root of the project
2. Add `ANTHROPIC_API_KEY=your_api_key` to the file

### Run API without docker

```bash
fastapi run api/main.py --proxy-headers --port 8080
```

## API usage

### Request

Post request to the `http://localhost:8080/red_flags` with the following body:

```json
{
  "content": "text to be analyzed"
}
```

### Response

Response will be in the following format:

```json
[
  {
    RedFlagId: [1/2/3/4],
    Phrase: [string],
    Explanation: [string]
  },
  ...
]
```

## Red flags

  - 1: Whataboutism
  - 2: Emotional Clickbait
  - 3: Trolling
  - 4: Polarization

## Docker

### Local

To build the docker image run:

```bash
docker build -t red_flags .
```

Then run to run the API:

```bash
docker run -p 8080:8080 red_flags
```

### Pull from Docker Hub

To pull the image from Docker Hub run:

```bash
docker pull yehorkhod/red_flags
```

Then run to run the API:

```bash
docker run -p 8080:8080 yehorkhod/red_flags
```
