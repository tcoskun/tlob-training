"""
LSTM Fiyat Tahmin Projesi - Ana Çalıştırma Dosyası
"""


from src.lstm_analyzer import LSTMAnalyzer
from src.lstm_portfolio_analyzer import LSTMPortfolioAnalyzer

def main():
    """
    Main function to run the LSTM model training and evaluation.
    """

    lstm_analyzer = LSTMAnalyzer()
    lstm_portfolio_analyzer = LSTMPortfolioAnalyzer()

    """ Tensorflow Analysis """
    lstm_analyzer.start_analysis();

    """ Tensorflow Portfolio Analysis """
    lstm_portfolio_analyzer.readModel()
    lstm_portfolio_analyzer.create_trading_decisions('momentum')
    lstm_portfolio_analyzer.analyze_decisions()
    lstm_portfolio_analyzer.run_backtest()
    
if __name__ == "__main__":
    main()