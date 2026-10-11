from app import create_app
from app.predictor import predictor

create_app()
if predictor.model is None:
    raise SystemExit('model loading failed')
print('loaded ok')
