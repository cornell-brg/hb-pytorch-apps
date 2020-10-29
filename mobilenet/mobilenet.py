import torch
import torchvision

def extra_arg_parser(parser):
    parser.add_argument('--lr', default=0.02, type=int,
                        help="learning rate")
    parser.add_argument('--momentum', default=0.9, type=int,
                        help="momentum")


if __name__ == "__main__":
    # Parse arguments
    args = utils.parse_model_args(extra_arg_parser)

    # Model & hyper-parameters
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    MOMENTUM = args.momentum

    model =  torchvision.models.mobilenet_v2()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    loss_func = nn.CrossEntropyLoss()

    # Data
    transforms = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.ImageNet(
        root='./data', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.ImageNet(
        root='./data', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    # Load pretrained model if necessary
    if args.load_model:
        model.load_state_dict(torch.load(args.model_filename))

    # Move model to HammerBlade if using HB
    if args.hammerblade:
        model.to(torch.device("hammerblade"))

    print(model)

    # Quit here if dry run
    if args.dry:
        exit(0)

    # Training
    if args.training:
        utils.train(model, trainloader, optimizer, loss_func, args)

    # Inference
    if args.inference:

        num_correct = [0]

        def collector(outputs, targets):
            pred = outputs.cpu().max(1)[1]
            num_correct[0] += pred.eq(targets.cpu().view_as(pred)).sum().item()

        utils.inference(model, testloader, loss_func, collector, args)

        num_correct = num_correct[0]
        test_accuracy = 100. * (num_correct / len(testloader.dataset))

        print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            num_correct,
            len(testloader.dataset),
            test_accuracy
        ))

    # Save model
    if args.save_model:
        utils.save_model(model, args.model_filename)
