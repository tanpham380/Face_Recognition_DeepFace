from functools import wraps
from flask import request, jsonify

from core.utils.static_variable import API_KEY

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        # print(f"Received API Key: {api_key}")
        # print(f"Expected API Key: {API_KEY}")
        if api_key and api_key == API_KEY:
            return f(*args, **kwargs)
        else:
            return jsonify({"message": "Invalid or missing API key"}), 403
    return decorated_function
