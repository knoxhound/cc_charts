# Import required libraries
import ccxt                             # Crypto exchange interactions
import pandas as pd                     # Data manipulation
import matplotlib.pyplot as plt         # Visualizations
from datetime import datetime           # Dates and times
import time                             # Delays, timing operations
from typing import Dict, Any
from pandas import DataFrame            # DataFrame import


class CryptoAnalyzer:
    def __init__(self, exchange_id='binanceus', api_key=None, api_secret=None):
        """
         Parameters:
        - exchange_id: str, default 'binanceus' - The exchange to connect to
        - api_key: Optional API key for authentication
        - api_secret: Optional API secret for authentication
        """
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
                'apiKey': api_key,
                'secret': api_secret
            })
            self.exchange.load_markets()
            print(f"Successfully connected to {exchange_id}")
        except Exception as e:
            print(f"Error connecting to {exchange_id}: {str(e)}")
            print("Attempting to connect to alternate exchanges...")
            for alt_exchange in ['binanceus', 'kraken', 'coinbasepro', 'kucoin']:
                if alt_exchange != exchange_id:
                    try:
                        self.exchange = getattr(ccxt, alt_exchange)({
                            'enableRateLimit': True
                        })
                        self.exchange.load_markets()
                        print(f"Successfully connected to {alt_exchange}")
                        break
                    except Exception as e:
                        print(f"Error connecting to {alt_exchange}: {str(e)}")

        self.order_book_data = {}
        self.market_data = {}

    def check_symbol_availability(self, symbol: str) -> str | None | Any:
        """
        Check if the symbol is available on the exchange and return the correct format
        """
        try:
            markets = self.exchange.markets
            # Try different common symbol formats
            variants = [
                symbol,
                symbol.replace('/', ''),
                symbol.replace('USDT', 'USD'),
                symbol.replace('/', '-'),
            ]

            for variant in variants:
                if variant in markets:
                    return variant

            # If none of the variants work, find a similar symbol
            similar_symbols = [s for s in markets.keys() if symbol.split('/')[0] in s]
            if similar_symbols:
                print(f"Symbol {symbol} not found. Using {similar_symbols[0]} instead.")
                return similar_symbols[0]

            raise Exception(f"Symbol {symbol} not available on this exchange")

        except Exception as e:
            print(f"Error checking symbol {symbol}: {str(e)}")
            return None

    def fetch_ohlcv_data(self, symbol: str, timeframe='1h', limit=500) -> DataFrame | None:
        """
        Fetch OHLCV data from the exchange
        """
        try:
            # Check and get correct symbol format
            formatted_symbol = self.check_symbol_availability(symbol)
            if not formatted_symbol:
                return None

            # Add retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
                    if ohlcv and len(ohlcv):
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        return df
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)  # Exponential backoff

            return None
        except Exception as e:
            print(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            return None

    def get_order_book_analysis(self, symbol: str, depth=100) -> dict[str, float | int | datetime] | None:
        """
        Analyze order book data including bid/ask volumes and imbalances
        """
        try:
            formatted_symbol = self.check_symbol_availability(symbol)
            if not formatted_symbol:
                return None

            order_book = self.exchange.fetch_order_book(formatted_symbol, depth)

            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                return None

            # Calculate bid and ask volumes
            bid_volume = sum(bid[1] for bid in order_book['bids'])
            ask_volume = sum(ask[1] for ask in order_book['asks'])

            # Calculate volume weighted average prices (VWAP)
            bid_vwap = sum(bid[0] * bid[1] for bid in order_book['bids']) / bid_volume if bid_volume > 0 else 0
            ask_vwap = sum(ask[0] * ask[1] for ask in order_book['asks']) / ask_volume if ask_volume > 0 else 0

            # Calculate imbalance
            total_volume = bid_volume + ask_volume
            bid_percentage = (bid_volume / total_volume * 100) if total_volume > 0 else 0

            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'bid_vwap': bid_vwap,
                'ask_vwap': ask_vwap,
                'bid_percentage': bid_percentage,
                'ask_percentage': 100 - bid_percentage,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {str(e)}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> DataFrame | None:

        # Calculate all technical indicators

        try:
            if df is None or df.empty:
                return None

            # Bollinger Bands
            df['middle_band'] = df['close'].rolling(window=20).mean()
            rolling_std = df['close'].rolling(window=20).std()
            df['upper_band'] = df['middle_band'] + (rolling_std * 2)
            df['lower_band'] = df['middle_band'] - (rolling_std * 2)

            # Stochastic Oscillator
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['%K'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
            df['%D'] = df['%K'].rolling(window=3).mean()

            # VWMA
            df['VWMA'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_spike'] = df['volume'] > (df['volume_ma'] * 2)

            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return None

    def plot_analysis(self, df: pd.DataFrame, symbol: str, order_book_data: Dict):
        """
        Create comprehensive visualization including order book data
        """
        try:
            if df is None or df.empty:
                return

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16),
                                                     gridspec_kw={'height_ratios': [3, 1, 1, 1]})

            # Plot 1: Price and Indicators
            ax1.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
            ax1.plot(df.index, df['upper_band'], label='Upper BB', color='gray', linestyle='--', alpha=0.5)
            ax1.plot(df.index, df['middle_band'], label='Middle BB', color='gray', linestyle='--', alpha=0.5)
            ax1.plot(df.index, df['lower_band'], label='Lower BB', color='gray', linestyle='--', alpha=0.5)
            ax1.plot(df.index, df['VWMA'], label='VWMA', color='orange', alpha=0.7)

            # Highlight volume spikes
            spike_dates = df[df['volume_spike']].index
            for date in spike_dates:
                ax1.axvline(x=date, color='red', alpha=0.2)

            ax1.set_title(f'{symbol} Technical Analysis - {self.exchange.name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Volume
            ax2.bar(df.index, df['volume'], color='blue', alpha=0.5)
            ax2.plot(df.index, df['volume_ma'], color='orange', label='Volume MA')
            ax2.set_title('Volume')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Stochastic Oscillator
            ax3.plot(df.index, df['%K'], label='%K', color='blue')
            ax3.plot(df.index, df['%D'], label='%D', color='orange')
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5)
            ax3.set_title('Stochastic Oscillator')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Order Book Analysis
            if order_book_data:
                labels = ['Bid Volume', 'Ask Volume']
                volumes = [order_book_data['bid_volume'], order_book_data['ask_volume']]
                colors = ['green', 'red']

                ax4.bar(labels, volumes, color=colors, alpha=0.6)
                ax4.set_title('Order Book Analysis')
                ax4.text(0, volumes[0], f'{order_book_data["bid_percentage"]:.1f}%',
                         ha='center', va='bottom')
                ax4.text(1, volumes[1], f'{order_book_data["ask_percentage"]:.1f}%',
                         ha='center', va='bottom')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'{symbol.replace("/", "_")}_{self.exchange.name}_analysis.png'
            plt.savefig(filename)
            plt.close()
            print(f"Saved analysis plot to {filename}")

        except Exception as e:
            print(f"Error creating visualization: {str(e)}")

    def save_data(self, df: pd.DataFrame, symbol: str, order_book_data: Dict):
        """
        Save both OHLCV and order book data to CSV
        """
        try:
            if df is None or df.empty:
                return None

            # Reset index to make timestamp a column
            df = df.reset_index()

            # Rename columns to include the symbol
            symbol_prefix = symbol.split('/')[0]  # Get 'BTC' from 'BTC/USDT'
            df = df.rename(columns={
                'open': f'{symbol_prefix}_open',
                'high': f'{symbol_prefix}_high',
                'low': f'{symbol_prefix}_low',
                'close': f'{symbol_prefix}_close',
                'volume': f'{symbol_prefix}_volume',
                'middle_band': f'{symbol_prefix}_BB_middle',
                'upper_band': f'{symbol_prefix}_BB_upper',
                'lower_band': f'{symbol_prefix}_BB_lower',
                '%K': f'{symbol_prefix}_stoch_K',
                '%D': f'{symbol_prefix}_stoch_D',
                'VWMA': f'{symbol_prefix}_VWMA',
                'volume_ma': f'{symbol_prefix}_volume_MA',
                'volume_spike': f'{symbol_prefix}_volume_spike'
            })

            # Add order book data to the last row
            if order_book_data:
                last_row = df.index[-1]
                df.loc[last_row, f'{symbol_prefix}_bid_volume'] = order_book_data['bid_volume']
                df.loc[last_row, f'{symbol_prefix}_ask_volume'] = order_book_data['ask_volume']
                df.loc[last_row, f'{symbol_prefix}_bid_vwap'] = order_book_data['bid_vwap']
                df.loc[last_row, f'{symbol_prefix}_ask_vwap'] = order_book_data['ask_vwap']
                df.loc[last_row, f'{symbol_prefix}_bid_percentage'] = order_book_data['bid_percentage']

            df['symbol'] = symbol
            df['exchange'] = self.exchange.name
            return df

        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return None


def main():
    # Initialize analyzer (trying multiple exchanges if needed)
    exchanges = ['binanceus', 'kraken', 'coinbasepro', 'kucoin']
    analyzer = None

    for exchange_id in exchanges:
        try:
            analyzer = CryptoAnalyzer(exchange_id=exchange_id)
            if hasattr(analyzer, 'exchange'):
                break
        except Exception as e:
            print(f"Failed to initialize {exchange_id}: {str(e)}")

    if not analyzer:
        print("Failed to connect to any exchange. Exiting.")
        return

    # Symbols to analyze
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
    all_data = []

    for symbol in symbols:
        print(f"Analyzing {symbol}...")

        # Fetch and analyze OHLCV data
        df = analyzer.fetch_ohlcv_data(symbol)
        if df is not None:
            df = analyzer.calculate_technical_indicators(df)

            # Get order book analysis
            order_book_data = analyzer.get_order_book_analysis(symbol)

            # Create visualization
            analyzer.plot_analysis(df, symbol, order_book_data)

            # Prepare data for CSV
            df = analyzer.save_data(df, symbol, order_book_data)
            if df is not None:
                all_data.append(df)

            # Avoid rate limiting
            time.sleep(analyzer.exchange.rateLimit / 1000)

    # Combine all data and export to CSV
    if all_data:
        # First, ensure all DataFrames have the same timestamp values
        base_timestamps = all_data[0]['timestamp'].values
        aligned_data = []

        for df in all_data:
            df_aligned = df[df['timestamp'].isin(base_timestamps)]
            aligned_data.append(df_aligned)

        # Merge all DataFrames on timestamp
        combined_df = aligned_data[0]
        for df in aligned_data[1:]:
            # Get the symbol prefix for the current DataFrame
            symbol_prefix = df['symbol'].iloc[0].split('/')[0]

            # Select columns that start with the symbol prefix and include timestamp
            symbol_cols = [col for col in df.columns if col.startswith(symbol_prefix) or col in ['timestamp']]
            df_subset = df[symbol_cols]

            # Merge with the combined DataFrame
            combined_df = pd.merge(combined_df, df_subset, on='timestamp', how='outer')

        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')

        # Organize columns
        # First get all columns and group them by cryptocurrency
        col_groups = {}
        for col in combined_df.columns:
            if col in ['timestamp', 'exchange']:
                continue
            for crypto in ['BTC', 'ETH', 'XRP']:
                if col.startswith(crypto) or col == f'symbol_{crypto}':
                    if crypto not in col_groups:
                        col_groups[crypto] = []
                    col_groups[crypto].append(col)

        # Reorder columns with timestamp first, then each crypto's data grouped together
        ordered_cols = ['timestamp']
        for crypto in ['BTC', 'ETH', 'XRP']:
            if crypto in col_groups:
                ordered_cols.extend(sorted(col_groups[crypto]))
        ordered_cols.append('exchange')

        # Reorder the columns in the DataFrame
        combined_df = combined_df[ordered_cols]

        # Export to CSV
        filename = f'crypto_analysis_{analyzer.exchange.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        combined_df.to_csv(filename, index=False)
        print(f"Analysis complete. Data saved to '{filename}'")

        # Also save individual CSV files for each cryptocurrency
        for symbol_data in all_data:
            symbol = symbol_data['symbol'].iloc[0].split('/')[0]
            filename = f'{symbol}_analysis_{analyzer.exchange.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            symbol_data.to_csv(filename, index=False)
            print(f"Individual {symbol} data saved to '{filename}'")
    else:
        print("No data was collected. Please check the error messages above.")


if __name__ == "__main__":
    main()