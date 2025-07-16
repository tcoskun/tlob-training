import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add TLOB library to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TLOB'))

from TLOB.models.tlob import TLOB
from TLOB.preprocessing.dataset import Dataset, DataModule
from TLOB.constants import DEVICE, N_LOB_LEVELS, LEN_LEVEL, LEN_ORDER
import TLOB.constants as cst


class TLOBIntegration:
    """Integration class for the real TLOB library"""
    
    def __init__(self, config):
        self.config = config
        self.device = DEVICE
        self.model = None
        self.data_module = None
        
    def prepare_data(self, data_path):
        """Prepare data for TLOB model"""
        # Load CSV data
        df = pd.read_csv(data_path, sep=';', decimal=',')
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        numeric_columns = [col for col in df.columns 
                          if any(x in col for x in ['Price', 'Volume', 'Ratio'])]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert to TLOB format
        # TLOB expects specific features based on LOBSTER format
        features = self._extract_tlob_features(df)
        
        # Create sequences
        seq_size = self.config.get('seq_size', 128)
        horizon = self.config.get('horizon', 10)
        
        # Split data
        train_split = self.config.get('data', {}).get('train_split', 0.8)
        val_split = self.config.get('data', {}).get('val_split', 0.1)
        test_split = self.config.get('data', {}).get('test_split', 0.1)
        
        train_size = int(train_split * len(features))
        val_size = int(val_split * len(features))
        
        train_features = features[:train_size]
        val_features = features[train_size:train_size + val_size]
        test_features = features[train_size + val_size:]
        
        print(f"📊 Data Split Information:")
        print(f"   Total features: {len(features):,}")
        print(f"   Train features: {len(train_features):,} ({train_split*100:.0f}%)")
        print(f"   Val features: {len(val_features):,} ({val_split*100:.0f}%)")
        print(f"   Test features: {len(test_features):,} ({test_split*100:.0f}%)")
        print(f"   Data source: {data_path}")
        
        # Create labels (simplified - you might want to implement proper labeling)
        train_labels = self._create_labels(train_features, horizon)
        val_labels = self._create_labels(val_features, horizon)
        test_labels = self._create_labels(test_features, horizon)
        
        # Truncate features and labels to the same length
        def match_length(feat, lab):
            min_len = min(len(feat), len(lab))
            # Ensure we have enough data for at least one sequence
            seq_size = self.config.get('seq_size', 128)
            usable_len = min_len - seq_size
            if usable_len <= 0:
                raise ValueError(f"Not enough data for sequence size {seq_size}. Need at least {seq_size + 1} samples.")
            return feat[:usable_len], lab[:usable_len]
        train_features, train_labels = match_length(train_features, train_labels)
        val_features, val_labels = match_length(val_features, val_labels)
        test_features, test_labels = match_length(test_features, test_labels)
        
        # Convert to tensors
        train_input = torch.tensor(train_features, dtype=torch.float32)
        val_input = torch.tensor(val_features, dtype=torch.float32)
        test_input = torch.tensor(test_features, dtype=torch.float32)
        
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_labels = torch.tensor(val_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        
        # Create datasets
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        test_set = Dataset(test_input, test_labels, seq_size)
        
        # Create data module
        self.data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=self.config.get('batch_size', 32),
            test_batch_size=self.config.get('batch_size', 32) * 4,
            num_workers=16
        )
        
        return train_input.shape[1]  # Return number of features
        
    def _extract_tlob_features(self, df):
        """Extract features in TLOB format"""
        # TLOB expects 47 features for LOBSTER data
        # We'll create a simplified version based on available data
        
        features = []
        
        # Calculate mid price from bid/ask prices
        bid_price = df['Level 1 Bid Price'].values
        ask_price = df['Level 1 Ask Price'].values
        mid_price = (bid_price + ask_price) / 2
        
        print(f"📊 Price Analysis:")
        print(f"   Bid Price range: {bid_price.min():.2f} - {bid_price.max():.2f}")
        print(f"   Ask Price range: {ask_price.min():.2f} - {ask_price.max():.2f}")
        print(f"   Mid Price range: {mid_price.min():.2f} - {mid_price.max():.2f}")
        print(f"   Sample mid prices: {mid_price[:5].tolist()}")
        
        # Create feature matrix
        for i in range(len(df)):
            feature_vector = []
            
            # Price features (10 levels)
            for level in range(1, 11):
                bid_price_col = f'Level {level} Bid Price'
                ask_price_col = f'Level {level} Ask Price'
                bid_size_col = f'Level {level} Bid Volume'
                ask_size_col = f'Level {level} Ask Volume'
                
                bid_price_val = df[bid_price_col].iloc[i] if bid_price_col in df.columns else (mid_price[i] - 0.1)
                ask_price_val = df[ask_price_col].iloc[i] if ask_price_col in df.columns else (mid_price[i] + 0.1)
                bid_size_val = df[bid_size_col].iloc[i] if bid_size_col in df.columns else 100
                ask_size_val = df[ask_size_col].iloc[i] if ask_size_col in df.columns else 100
                
                feature_vector.extend([bid_price_val, ask_price_val, bid_size_val, ask_size_val])
            
            # Order type (1 feature) - simplified
            feature_vector.append(1)  # Market order type (int)
            
            # Additional features to reach 47 total
            mid_val = mid_price[i]
            volume_val = df['Total Bid Volume'].iloc[i] if 'Total Bid Volume' in df.columns else 1000
            timestamp_val = i
            feature_vector.extend([
                mid_val,  # Mid price
                volume_val,  # Volume
                timestamp_val,  # Timestamp (index)
                # Add more features as needed to reach 47
            ])
            
            # Pad or truncate to exactly 47 features
            while len(feature_vector) < 47:
                feature_vector.append(0.0)
            feature_vector = feature_vector[:47]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_labels(self, features, horizon):
        """Create labels for price movement prediction"""
        labels = []
        
        for i in range(len(features) - horizon):
            # Get mid price from the correct feature index
            mid_price_index = 41  # 40 bid/ask features + 1 order type
            current_price = features[i][mid_price_index]  # Mid price feature
            future_price = features[i + horizon][mid_price_index]
            
            # If mid price is not available, calculate from bid/ask prices
            if current_price == 0 or np.isnan(current_price):
                bid_price = features[i][0]  # Level 1 Bid Price
                ask_price = features[i][1]  # Level 1 Ask Price
                current_price = (bid_price + ask_price) / 2
            
            if future_price == 0 or np.isnan(future_price):
                bid_price = features[i + horizon][0]  # Level 1 Bid Price
                ask_price = features[i + horizon][1]  # Level 1 Ask Price
                future_price = (bid_price + ask_price) / 2
            
            # Simple labeling: up, stable, down
            price_change = (future_price - current_price) / current_price
            
            if price_change > 0.001:  # 0.1% threshold
                labels.append(0)  # Up
            elif price_change < -0.001:
                labels.append(2)  # Down
            else:
                labels.append(1)  # Stable
        
        # Pad the end
        labels.extend([1] * horizon)
        
        return labels
    
    def create_model(self, num_features):
        """Create TLOB model"""
        self.model = TLOB(
            hidden_dim=self.config.get('hidden_dim', 128),
            num_layers=self.config.get('num_layers', 4),
            seq_size=self.config.get('seq_size', 128),
            num_features=num_features,
            num_heads=self.config.get('num_heads', 1),
            is_sin_emb=self.config.get('is_sin_emb', True),
            dataset_type="CUSTOM"  # Use custom instead of LOBSTER
        ).to(self.device)
        
        return self.model
    
    def train_model(self, max_epochs=10):
        """Train the TLOB model with early stopping"""
        if self.model is None or self.data_module is None:
            raise ValueError("Model and data must be prepared first")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('lr', 0.0001)
        )
        
        # Loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Early stopping parameters
        early_stopping_patience = self.config.get('early_stopping_patience', 3)
        min_delta = self.config.get('min_delta', 0.001)  # Minimum iyileşme eşiği
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_mae': [],
            'val_mse': [],
            'val_mape': [],
            'val_direction_acc': []
        }
        
        print(f"🚀 Starting training with early stopping (patience: {early_stopping_patience}, min_delta: {min_delta})")
        print(f"📊 Training for max {max_epochs} epochs with learning rate: {self.config.get('lr', 0.0001)}")
        
        for epoch in range(max_epochs):
            # Training
            self.model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_train += pred.eq(target.view_as(pred)).sum().item()
                total_train += target.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * correct_train / total_train
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * correct / total
            
            # Calculate validation metrics for this epoch
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_predictions.extend(pred.cpu().numpy().flatten())
                    val_targets.extend(target.cpu().numpy())
            
            # Calculate MAE, MSE, MAPE for validation set
            val_metrics = self._calculate_mae_mse_mape(np.array(val_predictions), np.array(val_targets))
            
            # Save to history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_acc'].append(train_accuracy)
            self.history['val_acc'].append(val_accuracy)
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_mape'].append(val_metrics['mape'])
            self.history['val_direction_acc'].append(val_metrics['direction_accuracy'])
            
            print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            print(f"  📊 Val Metrics: MAE: {val_metrics['mae']:.4f} | MSE: {val_metrics['mse']:.4f} | MAPE: {val_metrics['mape']:.2f}% | Dir Acc: {val_metrics['direction_accuracy']:.2f}%")
            
            # Early stopping logic with minimum improvement threshold
            improvement = best_val_loss - avg_val_loss
            if improvement > min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_tlob_model.pth')
                print(f"✅ New best validation loss: {best_val_loss:.4f} (improvement: {improvement:.4f}) - Model saved!")
                print(f"   🏆 Best Val Metrics: MAE: {val_metrics['mae']:.4f} | MSE: {val_metrics['mse']:.4f} | MAPE: {val_metrics['mape']:.2f}% | Dir Acc: {val_metrics['direction_accuracy']:.2f}%")
            else:
                patience_counter += 1
                print(f"⚠️  No significant improvement for {patience_counter}/{early_stopping_patience} epochs (improvement: {improvement:.4f}, threshold: {min_delta:.4f})")
                
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs!")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
        
        # Load the best model
        self.model.load_state_dict(torch.load('models/best_tlob_model.pth'))
        print(f"📥 Loaded best model with validation loss: {best_val_loss:.4f}")
        
        return self.history
    
    def evaluate_model(self):
        """Evaluate the model on test data with MAE, MSE, MAPE metrics"""
        if self.model is None or self.data_module is None:
            raise ValueError("Model must be trained first")
        
        self.model.eval()
        test_loader = self.data_module.test_dataloader()
        
        # Initialize metrics
        test_loss = 0
        correct = 0
        total = 0
        
        # For regression metrics, we need actual prices and predicted prices
        all_predictions = []
        all_targets = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        print("🧪 Evaluating model on test data...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                test_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Store predictions and targets for analysis
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate basic metrics
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate MAE, MSE, MAPE metrics
        metrics = self._calculate_mae_mse_mape(all_predictions, all_targets)
        
        # Print results
        print("\n" + "="*50)
        print("📊 TEST EVALUATION RESULTS")
        print("="*50)
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Correct Predictions: {correct}/{total}")
        print()
        print("📈 Core Metrics:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAPE (Classification Error): {metrics['mape']:.2f}%")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
        print()
        print("📊 Class Distribution:")
        print(f"  Predictions: {metrics['class_distribution']['predictions']}")
        print(f"  Targets: {metrics['class_distribution']['targets']}")
        print("="*50)
        
        return metrics
    
    def _calculate_mae_mse_mape(self, predictions, targets):
        """Calculate MAE, MSE, and MAPE metrics for classification problem"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # For classification problem, we'll calculate metrics based on:
        # 1. Direction accuracy (how well we predict up/down/stable)
        # 2. Simulated price changes for regression-like metrics
        
        # Convert predictions to price changes for regression metrics
        predicted_price_changes = []
        actual_price_changes = []
        
        # For each prediction, simulate the price change
        for pred, target in zip(predictions, targets):
            # Predicted price change based on direction
            if pred == 0:  # Up
                pred_change = 0.01  # +1%
            elif pred == 2:  # Down
                pred_change = -0.01  # -1%
            else:  # Stable
                pred_change = 0.0  # 0%
            
            # Actual price change based on target
            if target == 0:  # Up
                actual_change = 0.01  # +1%
            elif target == 2:  # Down
                actual_change = -0.01  # -1%
            else:  # Stable
                actual_change = 0.0  # 0%
            
            predicted_price_changes.append(pred_change)
            actual_price_changes.append(actual_change)
        
        predicted_price_changes = np.array(predicted_price_changes)
        actual_price_changes = np.array(actual_price_changes)
        
        # Calculate MAE and MSE
        mae = mean_absolute_error(actual_price_changes, predicted_price_changes)
        mse = mean_squared_error(actual_price_changes, predicted_price_changes)
        
        # Completely new MAPE calculation for classification
        # Instead of using simulated price changes, calculate MAPE based on classification errors
        # MAPE = (Number of misclassified samples / Total samples) * 100
        
        # Count misclassifications
        misclassifications = np.sum(predictions != targets)
        total_samples = len(predictions)
        
        # Calculate MAPE as classification error rate
        mape = (misclassifications / total_samples) * 100 if total_samples > 0 else 0.0
        
        # Additional metrics for classification
        direction_accuracy = np.mean(predictions == targets) * 100
        
        # Calculate class distribution for debugging
        unique_predictions, pred_counts = np.unique(predictions, return_counts=True)
        unique_targets, target_counts = np.unique(targets, return_counts=True)
        
        class_distribution = {
            'predictions': dict(zip(unique_predictions.astype(int), pred_counts.astype(int))),
            'targets': dict(zip(unique_targets.astype(int), target_counts.astype(int)))
        }
        
        return {
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'class_distribution': class_distribution,
            'predictions': predictions,
            'targets': targets
        }
    
    def predict(self, data):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(data_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            
        return prediction, probabilities.cpu().numpy()[0]
    
    def predict_days(self, initial_data, seq_size=64):
        """Predict next N days with mid price simulation"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get forecast days from config
        forecast_days = self.config.get('forecast_days', 10)
        
        predictions = []
        # Take the last sequence from the batch
        current_sequence = initial_data[-1]  # Shape: [seq_size, features]
        
        # Get the last known mid price from the mid price feature
        # TLOB format: 40 bid/ask prices (10 levels * 4 features) + 1 order type + 6 additional features
        # Mid price is at index 41 (after all bid/ask features and order type)
        mid_price_index = 41  # 40 bid/ask features + 1 order type
        last_mid_price = float(current_sequence[-1][mid_price_index])  # Last time step, mid price feature
        
        # If mid price is not available, try to calculate from bid/ask prices
        if last_mid_price == 0 or np.isnan(last_mid_price):
            # Calculate from Level 1 bid/ask prices (indices 0 and 1)
            bid_price = float(current_sequence[-1][0])  # Level 1 Bid Price
            ask_price = float(current_sequence[-1][1])  # Level 1 Ask Price
            last_mid_price = (bid_price + ask_price) / 2
        
        current_mid_price = last_mid_price
        
        print(f"🎯 Starting {forecast_days}-day mid price forecast...")
        print(f"   Initial mid price: {current_mid_price:.4f}")
        
        for day in range(forecast_days):
            # Make prediction
            pred, probs = self.predict(current_sequence)
            
            # Simulate mid price change based on prediction
            if pred == 0:  # Up
                price_change_pct = 0.01  # +1%
            elif pred == 2:  # Down
                price_change_pct = -0.01  # -1%
            else:  # Stable
                price_change_pct = 0.0  # 0%
            
            # Calculate new mid price
            new_mid_price = current_mid_price * (1 + price_change_pct)
            
            predictions.append({
                'day': day + 1,
                'prediction': pred,
                'probabilities': probs,
                'confidence': np.max(probs),
                'mid_price': new_mid_price,
                'price_change_pct': price_change_pct * 100
            })
            
            # Update current mid price for next iteration
            current_mid_price = new_mid_price
            
            # Update sequence for next prediction (simplified)
            # In practice, you'd need to update with actual market data
            current_sequence = np.roll(current_sequence, -1, axis=0)
            # Add some noise to simulate new data
            current_sequence[-1] += np.random.normal(0, 0.001, current_sequence.shape[1])
        
        print(f"✅ {forecast_days}-day mid price forecast completed!")
        print(f"   Final mid price: {current_mid_price:.4f}")
        
        return predictions
    
    def save_test_metrics(self, metrics, filename='results/test_metrics.json'):
        """Save test metrics to a JSON file"""
        import json
        import os
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to JSON serializable types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert all metrics to JSON serializable format
        metrics_to_save = convert_to_json_serializable(metrics)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"💾 Test metrics saved to {filename}") 