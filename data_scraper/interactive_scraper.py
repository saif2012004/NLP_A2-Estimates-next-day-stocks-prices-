"""
Interactive Financial Data Scraper
Allows user to input exchange and symbol for data collection
"""

from financial_data_scraper import FinancialDataScraper
import sys


def get_user_input():
    """Get user input for exchange and symbol"""
    print("Financial Data Scraper for FinTech Prediction")
    print("=" * 50)
    
    # Get exchange name
    exchange = input("Enter stock exchange name (e.g., NYSE, NASDAQ, PSX, Crypto): ").strip()
    
    # Get symbol
    symbol = input("Enter stock symbol or crypto ticker (e.g., AAPL, BTC-USD): ").strip().upper()
    
    # Get output format
    format_choice = input("Output format (csv/json) [default: csv]: ").strip().lower()
    if format_choice not in ['csv', 'json']:
        format_choice = 'csv'
    
    return exchange, symbol, format_choice


def main():
    """Main interactive function"""
    try:
        # Get user input
        exchange, symbol, output_format = get_user_input()
        
        print(f"\nProcessing: {symbol} on {exchange}")
        print("This may take a few moments...")
        
        # Initialize scraper
        scraper = FinancialDataScraper()
        
        # Scrape data
        dataset, filename = scraper.scrape_financial_data(exchange, symbol, output_format)
        
        print(f"\n✅ Data collection completed successfully!")
        print(f"📁 Data saved to: {filename}")
        print(f"📊 Total records: {len(dataset)}")
        
        # Ask if user wants to process another symbol
        another = input("\nWould you like to process another symbol? (y/n): ").strip().lower()
        if another == 'y':
            main()  # Recursive call for another round
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please try again with a valid symbol.")
        
        retry = input("Would you like to try again? (y/n): ").strip().lower()
        if retry == 'y':
            main()


if __name__ == "__main__":
    main()
