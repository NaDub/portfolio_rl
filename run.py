from factories.forecast_factory import forecast_factory
from factories.rl_factory import rl_factory
from cli import get_parser, load_config, select_toml

def main():
    args = get_parser()
    path = select_toml(args=args)
    config = load_config(path)

    if args.type == 'DL':
        forecast_factory(config=config, mode=args.mode)
    
    elif args.type == 'RL':
        rl_factory(config=config, mode=args.mode)

if __name__ == '__main__':
    main()
