# middleware.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class CustomHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Add custom headers to the request
        response = await call_next(request)
        response.headers['X-Custom-Header'] = 'This is a custom header'
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request details (just for example purposes)
        print(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        return response
