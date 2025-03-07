import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

app = FastAPI()

base_directory = os.getcwd()


def generate_directory_listing(folder_path: str, base_url: str) -> str:
    files = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            item += "/"
        files.append(f'<li><a href="{base_url}/{item}">{item}</a></li>')
    return f"""
    <html>
        <body>
            <h1>Directory listing for {folder_path}</h1>
            <ul>
                {"".join(files)}
            </ul>
        </body>
    </html>
    """


@app.get("/{path:path}")
async def serve_file_or_directory(path: str):
    full_path = os.path.join(base_directory, path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Path not found")

    if os.path.isdir(full_path):
        return HTMLResponse(content=generate_directory_listing(full_path, f"/{path}"))

    return FileResponse(full_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
