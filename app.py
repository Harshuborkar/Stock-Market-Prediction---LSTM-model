from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3, os, pandas as pd, feedparser
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import LSTMModel

app = Flask(__name__)
app.secret_key = 'your_secret_key'

if not os.path.exists('users.db'):
    conn = sqlite3.connect('users.db')

    c = conn.cursor()
    c.execute('''CREATE TABLE users (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return redirect(url_for('home')) if 'username' in session else redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=? AND password=?', (u, p))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = u
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid Credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        u, p, cp = request.form['username'], request.form['password'], request.form['confirm_password']
        if p != cp:
            return render_template('signup.html', error="Passwords do not match")
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=?', (u,))
        if c.fetchone():
            conn.close()
            return render_template('login.html', error="Username already exists")
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (u, p))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/analysis')
def analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('analysis.html')

def split_data(df):
    split_index = len(df) // 2
    return df[:split_index], df[split_index:]

def prepare_tensors(data):
    prices = data[['Close']].values.astype(float)
    X, y = [], []
    for i in range(10, len(prices)):
        X.append(prices[i-10:i])
        y.append(prices[i])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

def average_weights(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg = torch.stack(weights, dim=0).mean(dim=0)
        avg_weights.append(avg)
    return avg_weights


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'username' not in session:
        return redirect(url_for('login'))

    live_stocks = [
        "TATAPOWER.NS", "RELIANCE.NS", "SBIN.NS", "INFY.NS", "HDFCBANK.NS",
        "WIPRO.NS", "HINDUNILVR.NS", "TCS.NS", "AXISBANK.NS", "ICICIBANK.NS",
        "BAJFINANCE.NS", "ADANIENT.NS", "LT.NS", "ONGC.NS", "ITC.NS"
    ]

    live_data = []
    for symbol in live_stocks:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2d")
            if len(hist) < 2:
                continue
            latest = hist.iloc[-1]
            prev = hist.iloc[-2]
            price = round(latest["Close"], 2)
            change = round(price - prev["Close"], 2)
            percent = round((change / prev["Close"]) * 100, 2)
            live_data.append({
                "name": symbol.replace(".NS", ""),
                "price": price,
                "change": change,
                "percent": percent,
                "direction": "up" if change > 0 else "down"
            })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            df['MA10'] = df['Close'].rolling(window=10).mean()

            # Federated Learning Simulation
            df1, df2 = split_data(df)
            X1, y1 = prepare_tensors(df1)
            X2, y2 = prepare_tensors(df2)

            model = LSTMModel()
            criterion = nn.MSELoss()

            weights_list = []
            for X, y in [(X1, y1), (X2, y2)]:
                local_model = LSTMModel()
                local_model.load_state_dict(model.state_dict())
                optimizer = optim.Adam(local_model.parameters(), lr=0.001)

                local_model.train()
                for epoch in range(3):  # Fewer epochs for speed
                    optimizer.zero_grad()
                    output = local_model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()

                weights_list.append([param.data.clone() for param in local_model.parameters()])

            avg_weights = average_weights(weights_list)
            with torch.no_grad():
                for param, avg in zip(model.parameters(), avg_weights):
                    param.data.copy_(avg)

            model.eval()
            with torch.no_grad():
                prediction_input = torch.tensor(df[['Close']].values[-10:], dtype=torch.float32).unsqueeze(0)
                predicted_price = model(prediction_input).item()

            # Visualizations
            candlestick = go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            candlestick.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')

            line_chart = go.Figure()
            line_chart.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
            line_chart.update_layout(title='Stock Close Price Over Time', xaxis_title='Date', yaxis_title='Close')

            ma_chart = go.Figure()
            ma_chart.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
            ma_chart.add_trace(go.Scatter(x=df['Date'], y=df['MA10'], name='MA10'))
            ma_chart.update_layout(title='Moving Average Chart', xaxis_title='Date', yaxis_title='Price')

            latest_close = df['Close'].iloc[-1]
            ma10_latest = df['MA10'].iloc[-1]
            recommendation = "Buy" if predicted_price > latest_close else "Hold" if abs(predicted_price - latest_close) < 1 else "Sell"

            return render_template('prediction.html',
                                   live_data=live_data,
                                   candlestick=candlestick.to_html(full_html=False),
                                   line_chart=line_chart.to_html(full_html=False),
                                   ma_chart=ma_chart.to_html(full_html=False),
                                   recommendation=recommendation,
                                   latest_close=latest_close,
                                   predicted=round(predicted_price, 2),
                                   ma10=ma10_latest,
                                   datetime=datetime)

    return render_template('prediction.html', live_data=live_data, datetime=datetime)

@app.route('/market_news')
def market_news():
    if 'username' not in session:
        return redirect(url_for('login'))
    feed = feedparser.parse('https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms')
    news_items = [{'title': e.title, 'url': e.link, 'summary': e.summary, 'published': e.published} for e in feed.entries[:10]]
    return render_template('market_news.html', news=news_items)

@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
