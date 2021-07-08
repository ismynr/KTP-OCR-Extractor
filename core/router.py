import core.application as model

class Router:
    @staticmethod
    def run(app):
        @app.route('/')
        def home():
            return {
                'message':'API Using Flask Python For KTP Extractor & KTP Validation With Tesseract OCR',
            }

        @app.route('/api/extract_ktp', methods=['POST'])
        def extract():
            return model.extract_ktp()    