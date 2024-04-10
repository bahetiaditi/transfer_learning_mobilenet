def train_and_validate(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=15):
    train_losses, val_losses, avg_ious, avg_dices = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        val_running_loss, iou_scores, dice_scores = 0.0, [], []
        with torch.no_grad():
            for images, masks in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item()


                iou_scores.append(calc_iou(outputs, masks))
                dice_scores.append(calc_dice(outputs, masks))

        val_loss = val_running_loss / len(val_dataloader)
        val_losses.append(val_loss)

        avg_iou = torch.tensor(iou_scores).mean().item()
        avg_dice = torch.tensor(dice_scores).mean().item()
        avg_ious.append(avg_iou)
        avg_dices.append(avg_dice)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f}')
        visualize_prediction(images.cpu(), masks.cpu(), outputs.cpu(), epoch, num_epochs)


    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(avg_ious, label='Average IoU')
    plt.plot(avg_dices, label='Average Dice')
    plt.title('IoU and Dice Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_model(model, criterion, test_dataloader):
    model.eval()
    test_loss = 0.0
    iou_scores, dice_scores = [], []

    with torch.no_grad():
        for images, masks in test_dataloader:
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()


            iou_scores.append(calc_iou(outputs, masks))
            dice_scores.append(calc_dice(outputs, masks))


            if len(iou_scores) <= 1:
                visualize_test_prediction(images.cpu(), masks.cpu(), outputs.cpu())

    test_loss /= len(test_dataloader)
    avg_iou = torch.tensor(iou_scores).mean().item()
    avg_dice = torch.tensor(dice_scores).mean().item()

    print(f'Test Loss: {test_loss:.4f}, Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f}')