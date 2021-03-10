import logging
# must run makefile.bat to download kaggle files first!
# Then run this to build to features set

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    from src.features import build_features
    build_features.main()