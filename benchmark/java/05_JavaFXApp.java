package com.example.wisetok.ui.fx;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.util.Callback;
import java.io.IOException;

/**
 * JavaFX desktop application for tokenizer visualization and training.
 * Demonstrates Application lifecycle, FXML, observable lists, table view binding, and event handlers.
 */
public class TokenizerFXApp extends Application {

    @Override
    public void start(Stage primaryStage) throws IOException {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/main.fxml"));
        BorderPane root = loader.load();

        TokenizerController controller = loader.getController();
        controller.initialize(primaryStage);

        Scene scene = new Scene(root, 1024, 768);
        primaryStage.setTitle("WiseTok Tokenizer");
        primaryStage.setScene(scene);
        primaryStage.setOnCloseRequest(e -> controller.onApplicationClose());
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}

class TokenizerController {

    @FXML
    private TableView<TokenTableRow> tokenTable;

    @FXML
    private TableColumn<TokenTableRow, String> tokenColumn;

    @FXML
    private TableColumn<TokenTableRow, Integer> frequencyColumn;

    @FXML
    private TextField inputTextField;

    @FXML
    private TextArea outputTextArea;

    @FXML
    private Label statusLabel;

    @FXML
    private ProgressBar trainingProgressBar;

    @FXML
    private Button trainButton;

    @FXML
    private Button encodeButton;

    @FXML
    private ComboBox<String> patternComboBox;

    @FXML
    private Spinner<Integer> mergeCountSpinner;

    private ObservableList<TokenTableRow> tableData;

    @FXML
    public void initialize() {
        // Setup table columns
        tokenColumn.setCellValueFactory(new PropertyValueFactory<>("token"));
        frequencyColumn.setCellValueFactory(new PropertyValueFactory<>("frequency"));

        // Initialize table data
        tableData = FXCollections.observableArrayList();
        tokenTable.setItems(tableData);

        // Setup pattern combo box
        ObservableList<String> patterns = FXCollections.observableArrayList(
            "GPT-4 Pattern", "Custom Regex", "Simple Byte"
        );
        patternComboBox.setItems(patterns);
        patternComboBox.setValue("GPT-4 Pattern");

        // Setup merge count spinner
        SpinnerValueFactory<Integer> valueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(
            1000, 100000, 50000, 1000
        );
        mergeCountSpinner.setValueFactory(valueFactory);

        // Bind button states
        trainButton.disableProperty().bind(
            Bindings.or(
                inputTextField.textProperty().isEmpty(),
                trainingProgressBar.visibleProperty()
            )
        );

        encodeButton.disableProperty().bind(
            Bindings.isEmpty(tableData)
        );

        // Setup table selection listener
        tokenTable.getSelectionModel().selectedItemProperty().addListener(
            (obs, oldVal, newVal) -> {
                if (newVal != null) {
                    statusLabel.setText("Selected: " + newVal.getToken() +
                        " (freq: " + newVal.getFrequency() + ")");
                }
            }
        );
    }

    public void initialize(Stage primaryStage) {
        initialize();
    }

    @FXML
    protected void onTrainButtonClick(ActionEvent event) {
        String input = inputTextField.getText();
        int merges = mergeCountSpinner.getValue();

        trainingProgressBar.setVisible(true);
        trainButton.setDisable(true);

        // Simulate asynchronous training
        Thread trainThread = new Thread(() -> {
            try {
                simulateTraining(input, merges);
                Platform.runLater(this::onTrainingComplete);
            } catch (InterruptedException e) {
                Platform.runLater(() -> statusLabel.setText("Training interrupted"));
                Thread.currentThread().interrupt();
            }
        });
        trainThread.setDaemon(false);
        trainThread.start();
    }

    @FXML
    protected void onEncodeButtonClick(ActionEvent event) {
        String text = inputTextField.getText();
        if (text.isEmpty()) {
            statusLabel.setText("Enter text to encode");
            return;
        }

        try {
            // Simulate encoding
            StringBuilder result = new StringBuilder();
            for (char c : text.toCharArray()) {
                result.append((int) c).append(" ");
            }
            outputTextArea.setText(result.toString().trim());
            statusLabel.setText("Encoded successfully");
        } catch (Exception e) {
            statusLabel.setText("Encoding failed: " + e.getMessage());
        }
    }

    @FXML
    protected void onClearButtonClick(ActionEvent event) {
        inputTextField.clear();
        outputTextArea.clear();
        tableData.clear();
        statusLabel.setText("Cleared");
    }

    @FXML
    protected void onExportButtonClick(ActionEvent event) {
        // File save dialog
        javafx.stage.FileChooser fileChooser = new javafx.stage.FileChooser();
        fileChooser.setTitle("Export Tokenizer");
        fileChooser.getExtensionFilters().add(
            new javafx.stage.FileChooser.ExtensionFilter("JSON Files", "*.json")
        );
        // File file = fileChooser.showSaveDialog(trainButton.getScene().getWindow());
        statusLabel.setText("Export feature not implemented");
    }

    private void simulateTraining(String input, int merges) throws InterruptedException {
        for (int i = 0; i <= merges; i += 5000) {
            double progress = (double) i / merges;
            Platform.runLater(() -> {
                trainingProgressBar.setProgress(progress);
                statusLabel.setText("Training: " + (int)(progress * 100) + "%");
            });
            Thread.sleep(500);
        }
    }

    private void onTrainingComplete() {
        trainingProgressBar.setVisible(false);
        trainingProgressBar.setProgress(1.0);
        trainButton.setDisable(false);
        statusLabel.setText("Training complete");

        // Add sample data to table
        tableData.addAll(
            new TokenTableRow("hello", 1250),
            new TokenTableRow("world", 980),
            new TokenTableRow("token", 750),
            new TokenTableRow("merge", 650),
            new TokenTableRow("pair", 520)
        );
    }

    public void onApplicationClose() {
        // Cleanup resources
    }
}

class TokenTableRow {
    private final SimpleStringProperty token;
    private final SimpleIntegerProperty frequency;

    public TokenTableRow(String token, Integer frequency) {
        this.token = new SimpleStringProperty(token);
        this.frequency = new SimpleIntegerProperty(frequency);
    }

    public String getToken() {
        return token.get();
    }

    public void setToken(String token) {
        this.token.set(token);
    }

    public Integer getFrequency() {
        return frequency.get();
    }

    public void setFrequency(Integer frequency) {
        this.frequency.set(frequency);
    }
}
