"""
Small Cap / Affordable Stock Universe
Expanded ticker list for small capital accounts ($2-$25 price range)
"""

# ==============================================================================
# SMALL CAP STOCK UNIVERSE (Organized by Sector)
# ==============================================================================

# Technology & Software (Under $25)
TECH_SMALL_CAPS = [
    'SOFI', 'HOOD', 'SNAP', 'PINS', 'RBLX', 'U', 'DKNG', 'OPEN',
    'PLBY', 'SKLZ', 'WISH', 'CLOV', 'ROOT', 'LMND', 'UPST',
    'AFRM', 'SQ', 'PYPL', 'ROKU', 'SPOT', 'TWLO', 'ZM', 'DOCU',
    'CRWD', 'NET', 'DDOG', 'SNOW', 'MDB', 'ESTC', 'BILL'
]

# EV & Clean Energy (Under $25)
EV_CLEAN_ENERGY = [
    'RIVN', 'LCID', 'NKLA', 'FSR', 'RIDE', 'GOEV', 'WKHS',
    'PLUG', 'FCEL', 'BLDP', 'BE', 'RUN', 'ENPH', 'SEDG',
    'CHPT', 'BLNK', 'EVGO', 'QS', 'STEM', 'CLSK'
]

# Crypto & Fintech (Under $25)
CRYPTO_FINTECH = [
    'COIN', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF', 'ARBK',
    'MSTR', 'SQ', 'SOFI', 'HOOD', 'AFRM', 'UPST', 'LC',
    'NU', 'PAYO', 'STNE', 'PAGS'
]

# Telecom & Communications (Under $25)
TELECOM = [
    'NOK', 'ERIC', 'T', 'VZ', 'TMUS', 'LUMN', 'SATS'
]

# Retail & Consumer (Under $25)
RETAIL_CONSUMER = [
    'AMC', 'GME', 'BBBY', 'EXPR', 'KOSS', 'BB', 'NAKD',
    'WISH', 'PRPL', 'BYND', 'OATLY', 'TTCF', 'APPH'
]

# Travel & Hospitality (Under $25)
TRAVEL = [
    'AAL', 'UAL', 'DAL', 'LUV', 'JBLU', 'SAVE', 'ALK',
    'CCL', 'RCL', 'NCLH', 'EXPE', 'BKNG', 'ABNB', 'LYFT', 'UBER'
]

# Cannabis (Under $25)
CANNABIS = [
    'TLRY', 'CGC', 'ACB', 'SNDL', 'HEXO', 'OGI', 'CRON',
    'CURLF', 'GTBIF', 'TCNNF', 'CRLBF'
]

# Biotech & Healthcare (Under $25)
BIOTECH = [
    'MRNA', 'BNTX', 'NVAX', 'VXRT', 'INO', 'OCGN', 'SRNE',
    'ATOS', 'GEVO', 'AMRN', 'HGEN', 'VBIV', 'CODX'
]

# Penny Stocks with Volume (Under $10)
PENNY_STOCKS = [
    'SNDL', 'NOK', 'BB', 'NAKD', 'EXPR', 'KOSS', 'CLOV',
    'WKHS', 'RIDE', 'GOEV', 'GNUS', 'BNGO', 'ZOM', 'OCGN',
    'PROG', 'CEI', 'MULN', 'BBIG', 'ATER', 'RDBX'
]

# Meme Stocks (High Volume)
MEME_STOCKS = [
    'GME', 'AMC', 'BB', 'BBBY', 'NOK', 'KOSS', 'EXPR',
    'CLOV', 'WISH', 'WKHS', 'RIDE', 'SOFI', 'HOOD'
]

# Chinese ADRs (Under $25)
CHINESE_ADRS = [
    'NIO', 'XPEV', 'LI', 'BABA', 'JD', 'PDD', 'BIDU',
    'IQ', 'BILI', 'DIDI', 'GRAB', 'BEKE', 'TME'
]

# Banking & Finance (Under $25)
BANKING = [
    'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'ALLY',
    'SOFI', 'LC', 'UPST', 'AFRM'
]

# Energy & Oil (Under $25)
ENERGY = [
    'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'OXY', 'DVN',
    'MRO', 'APA', 'FANG', 'CLR', 'MPC', 'VLO', 'PSX'
]

# ==============================================================================
# COMBINED UNIVERSE
# ==============================================================================

def get_small_cap_universe(max_price=25.0, include_penny=True, include_meme=True):
    """
    Get expanded universe of affordable stocks
    
    Args:
        max_price: Maximum stock price to include
        include_penny: Include penny stocks (< $10)
        include_meme: Include meme stocks (high volatility)
    
    Returns:
        List of unique ticker symbols
    """
    universe = []
    
    # Core sectors (always included)
    universe.extend(TECH_SMALL_CAPS)
    universe.extend(EV_CLEAN_ENERGY)
    universe.extend(CRYPTO_FINTECH)
    universe.extend(TELECOM)
    universe.extend(RETAIL_CONSUMER)
    universe.extend(TRAVEL)
    universe.extend(CANNABIS)
    universe.extend(BIOTECH)
    universe.extend(CHINESE_ADRS)
    universe.extend(BANKING)
    universe.extend(ENERGY)
    
    # Optional sectors
    if include_penny:
        universe.extend(PENNY_STOCKS)
    
    if include_meme:
        universe.extend(MEME_STOCKS)
    
    # Remove duplicates and return
    return list(set(universe))


# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

# Conservative: Focus on established small caps
CONSERVATIVE_UNIVERSE = list(set(
    TECH_SMALL_CAPS + TELECOM + BANKING + ENERGY
))

# Aggressive: Include high-volatility plays
AGGRESSIVE_UNIVERSE = get_small_cap_universe(
    max_price=25.0,
    include_penny=True,
    include_meme=True
)

# Crypto-Focused: Crypto and related stocks
CRYPTO_UNIVERSE = list(set(
    CRYPTO_FINTECH + MEME_STOCKS + ['TSLA', 'SQ', 'PYPL']
))

# EV-Focused: Electric vehicles and clean energy
EV_UNIVERSE = list(set(
    EV_CLEAN_ENERGY + CHINESE_ADRS + ['TSLA', 'F', 'GM']
))

# Default: Balanced mix
DEFAULT_UNIVERSE = get_small_cap_universe(
    max_price=25.0,
    include_penny=False,  # Exclude very risky penny stocks
    include_meme=True     # Include meme stocks for volume
)

# ==============================================================================
# USAGE
# ==============================================================================

if __name__ == "__main__":
    pass  # print(f"Conservative Universe: {len(CONSERVATIVE_UNIVERSE))} tickers")
    pass  # print(f"Aggressive Universe: {len(AGGRESSIVE_UNIVERSE))} tickers")
    pass  # print(f"Crypto Universe: {len(CRYPTO_UNIVERSE))} tickers")
    pass  # print(f"EV Universe: {len(EV_UNIVERSE))} tickers")
    pass  # print(f"Default Universe: {len(DEFAULT_UNIVERSE))} tickers")
    
    print(f"\nDefault Universe Tickers:")
    print(sorted(DEFAULT_UNIVERSE))
