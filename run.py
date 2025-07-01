from cli import get_args, load_config, select_toml
from usecases.rl_case import RLUseCase
from usecases.forecast_case import DLUseCase

def main():
    args = get_args()
    path = select_toml(args=args)
    config = load_config(path)

    usecase_cls = {
        'RL': RLUseCase,
        'DL': DLUseCase
    }.get(args.type)

    if usecase_cls is None:
        raise ValueError(f"Unknown type '{args.type}', expected RL or DL")

    usecase = usecase_cls(config)

    if args.mode == 'train':
        usecase.train()
    elif args.mode == 'test':
        usecase.test()
    else:
        raise ValueError(f"Unknown mode '{args.mode}', expected train or test")

if __name__ == "__main__":
    main()