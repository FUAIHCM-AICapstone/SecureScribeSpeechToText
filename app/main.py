import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

from app.api import api_router
from app.core.config import settings
from app.core.vault_loader import load_config_from_api_v2

load_config_from_api_v2()


def custom_generate_unique_id(route: APIRoute) -> str:
    """
    Custom function to generate unique operation IDs for OpenAPI schema.
    This creates cleaner method names for generated client code.
    """
    if route.tags:
        return f"{route.tags[0]}_{route.name}"
    return route.name


def custom_openapi():
    """
    Custom OpenAPI schema generator with additional metadata and extensions.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",
        "altText": "SecureScribe API Logo",
    }

    openapi_schema["servers"] = [
        {"url": "http://localhost:9999", "description": "Development server"},
        {"url": "https://s2t.wc504.io.vn", "description": "Production server"},
    ]

    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Authorization header using the Bearer scheme.",
        }
    }

    openapi_schema["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return openapi_schema


app = FastAPI(
    title="SecureScribeBE",
    version="1.0.0",
    contact={
        "name": "SecureScribe Team",
        "email": "support@securescribe.com",
    },
    license_info={
        "name": "MIT",
    },
    generate_unique_id_function=custom_generate_unique_id,
)


@app.middleware("http")
async def log_requests(request, call_next):
    print(f"[REQUEST] {request.method} {request.url}")
    response = await call_next(request)
    print(f"[RESPONSE] {response}")
    return response


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=(["*"]),
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(api_router)


@app.get("/chat/css", response_class=FileResponse)
def serve_chat_css():
    """Serve chat_bubble.css file directly"""
    file_path = os.path.join("app", "static", "chat_bubble.css")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/css")
    else:
        raise HTTPException(status_code=404, detail="chat_bubble.css not found")


@app.get("/chat/js", response_class=FileResponse)
def serve_chat_js():
    """Serve chat_bubble.js file directly"""
    file_path = os.path.join("app", "static", "chat_bubble.js")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="chat_bubble.js not found")


@app.get("/chat")
def serve_chat_app():
    """Serve the complete chat application"""
    file_path = os.path.join("app", "static", "chat_bubble.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="chat_bubble.html not found")


@app.get("/health/redis")
def health_redis() -> Dict[str, Any]:
    """
    Redis health check endpoint
    """
    try:
        from app.utils.redis import get_redis_client

        redis_client = get_redis_client()

        redis_client.ping()

        info = redis_client.info()
        memory_info = redis_client.info("memory")

        return {
            "status": "connected",
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "db": settings.REDIS_DB_S2T,
            "version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "connected_clients": info.get("connected_clients", 0),
            "memory_used": memory_info.get("used_memory_human", "unknown"),
            "memory_peak": memory_info.get("used_memory_peak_human", "unknown"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "disconnected",
                "error": str(e),
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
            },
        )
