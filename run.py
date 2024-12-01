from app import create_app

app = create_app()

app.secret_key = '1234'

if __name__ == '__main__':
    app.run(debug=True)
