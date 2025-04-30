from flask import Flask
from flask_cors import CORS
from config.settings import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(config_class)

    # Registrar blueprints
    from app.routes.chat_routes import chat_blueprint
    from app.routes.document_routes import document_blueprint
    app.register_blueprint(chat_blueprint)
    app.register_blueprint(document_blueprint)

    return app