from fastapi import FastAPI

app = FastAPI()


@app.get("/score")
def score(value: int) -> dict[str, int]:
    raise NotImplementedError("Implement the endpoint contract")
