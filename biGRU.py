import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Assuming BiGRUModel and EmbeddingDataset are defined as in previous examples

# MLP Model Class for Scene Prediction
class ScenePredictionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScenePredictionMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Modified Model Trainer Class for Integrated Training
class IntegratedModelTrainer:
    def __init__(self, audio_model, video_model, mlp_model, data_loader, learning_rate=0.001, num_epochs=5):
        self.audio_model = audio_model
        self.video_model = video_model
        self.mlp_model = mlp_model
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self._init_optimizer()

    def _init_optimizer(self):
        # Include all parameters initially
        self.optimizer = torch.optim.Adam(
            list(self.audio_model.parameters()) + 
            list(self.video_model.parameters()) + 
            list(self.mlp_model.parameters()), 
            lr=self.learning_rate
        )

    def toggle_biGRU_trainability(self, trainable):
        # Freeze or unfreeze BiGRU models
        for param in self.audio_model.parameters():
            param.requires_grad = trainable
        for param in self.video_model.parameters():
            param.requires_grad = trainable

        # Reinitialize the optimizer with only MLP parameters if freezing BiGRU
        if not trainable:
            self.optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            for audio_embeddings, video_embeddings, labels in self.data_loader:
                audio_output = self.audio_model(audio_embeddings)
                video_output = self.video_model(video_embeddings)

                combined_output = torch.cat((audio_output, video_output), dim=1)
                scene_predictions = self.mlp_model(combined_output)

                loss = self.criterion(scene_predictions, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
class AudioModelTrainer:
    def __init__(self, audio_model, data_loader, learning_rate=0.001, num_epochs=5):
        self.audio_model = audio_model
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.audio_model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            for inputs, labels in self.data_loader:  # Assuming inputs are audio embeddings and labels
                outputs = self.audio_model(inputs)  # Get predictions from the audio model
                loss = self.criterion(outputs, labels)  # Calculate loss
                
                # Backpropagation
                self.optimizer.zero_grad()  # Clear gradients for this training step
                loss.backward()  # Backpropagation, compute gradients
                self.optimizer.step()  # Apply gradients
                
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
class AudioBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioBiGRU, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.bigru(x)  # out: batch, seq, hidden*2
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out