import os
os.chdir(r'C:\Users\User\OneDrive\Desktop\New folder\CNN Project')
from app import create_app

app = create_app()
with app.app_context():
    from app.predictor import predictor
    print(f'Model loaded: {predictor.model is not None}')
    print(f'Classes: {len(predictor.class_indices)}')
    if predictor.model:
        print(f'Model input shape: {predictor.model.input_shape}')
        print('Model type:', type(predictor.model).__name__)
