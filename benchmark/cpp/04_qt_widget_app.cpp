#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QSlider>
#include <QSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QHeaderView>

class CustomWidget : public QWidget {
    Q_OBJECT

private:
    QPushButton* btn_submit;
    QPushButton* btn_clear;
    QLineEdit* input_field;
    QTextEdit* output_area;
    QSlider* slider_intensity;
    QSpinBox* spin_value;
    QComboBox* combo_mode;
    QCheckBox* check_verbose;
    QLabel* status_label;

public:
    CustomWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setupUI();
        connectSignals();
    }

private slots:
    void onSubmitClicked() {
        QString text = input_field->text();
        if (text.isEmpty()) {
            QMessageBox::warning(this, "Input Error", "Please enter some text");
            return;
        }

        output_area->append(QString("Processed: ") + text);
        output_area->append(QString("Mode: ") + combo_mode->currentText());
        output_area->append(QString("Intensity: ") + QString::number(slider_intensity->value()));

        if (check_verbose->isChecked()) {
            output_area->append("Verbose mode enabled");
        }

        status_label->setText("Status: Ready");
    }

    void onClearClicked() {
        input_field->clear();
        output_area->clear();
        slider_intensity->setValue(50);
        spin_value->setValue(0);
        combo_mode->setCurrentIndex(0);
        check_verbose->setChecked(false);
        status_label->setText("Status: Cleared");
    }

    void onSliderChanged(int value) {
        spin_value->setValue(value);
        status_label->setText(QString("Slider value: %1").arg(value));
    }

    void onModeChanged(const QString& mode) {
        output_area->append(QString("Mode changed to: ") + mode);
    }

    void onVerboseToggled(bool checked) {
        if (checked) {
            output_area->append("[VERBOSE] Verbose logging enabled");
        }
    }

private:
    void setupUI() {
        QVBoxLayout* main_layout = new QVBoxLayout(this);

        QLabel* title_label = new QLabel("Qt Signal/Slot Demo");
        QFont title_font = title_label->font();
        title_font.setPointSize(14);
        title_font.setBold(true);
        title_label->setFont(title_font);
        main_layout->addWidget(title_label);

        QHBoxLayout* input_layout = new QHBoxLayout();
        input_layout->addWidget(new QLabel("Input:"));
        input_field = new QLineEdit();
        input_field->setPlaceholderText("Enter text here");
        input_layout->addWidget(input_field);
        main_layout->addLayout(input_layout);

        QHBoxLayout* combo_layout = new QHBoxLayout();
        combo_layout->addWidget(new QLabel("Mode:"));
        combo_mode = new QComboBox();
        combo_mode->addItem("Normal");
        combo_mode->addItem("Fast");
        combo_mode->addItem("Precise");
        combo_mode->addItem("Debug");
        combo_layout->addWidget(combo_mode);
        main_layout->addLayout(combo_layout);

        QHBoxLayout* slider_layout = new QHBoxLayout();
        slider_layout->addWidget(new QLabel("Intensity:"));
        slider_intensity = new QSlider(Qt::Horizontal);
        slider_intensity->setMinimum(0);
        slider_intensity->setMaximum(100);
        slider_intensity->setValue(50);
        slider_layout->addWidget(slider_intensity);
        spin_value = new QSpinBox();
        spin_value->setMinimum(0);
        spin_value->setMaximum(100);
        spin_value->setValue(50);
        slider_layout->addWidget(spin_value);
        main_layout->addLayout(slider_layout);

        check_verbose = new QCheckBox("Verbose Output");
        main_layout->addWidget(check_verbose);

        output_area = new QTextEdit();
        output_area->setReadOnly(true);
        output_area->setPlaceholderText("Output will appear here");
        main_layout->addWidget(new QLabel("Output:"));
        main_layout->addWidget(output_area);

        QHBoxLayout* button_layout = new QHBoxLayout();
        btn_submit = new QPushButton("Submit");
        btn_clear = new QPushButton("Clear");
        button_layout->addWidget(btn_submit);
        button_layout->addWidget(btn_clear);
        main_layout->addLayout(button_layout);

        status_label = new QLabel("Status: Ready");
        status_label->setStyleSheet("QLabel { color: blue; }");
        main_layout->addWidget(status_label);

        setLayout(main_layout);
        setWindowTitle("Qt Widget Application");
        resize(600, 500);
    }

    void connectSignals() {
        connect(btn_submit, &QPushButton::clicked, this, &CustomWidget::onSubmitClicked);
        connect(btn_clear, &QPushButton::clicked, this, &CustomWidget::onClearClicked);
        connect(slider_intensity, &QSlider::valueChanged, this, &CustomWidget::onSliderChanged);
        connect(combo_mode, QOverload<const QString&>::of(&QComboBox::currentTextChanged),
                this, &CustomWidget::onModeChanged);
        connect(check_verbose, &QCheckBox::toggled, this, &CustomWidget::onVerboseToggled);
    }
};

class MainWindow : public QMainWindow {
    Q_OBJECT

private:
    CustomWidget* central_widget;

public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        central_widget = new CustomWidget();
        setCentralWidget(central_widget);
        setWindowTitle("Qt Application - Main Window");
        resize(700, 600);
    }
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    MainWindow window;
    window.show();

    return app.exec();
}

#include "04_qt_widget_app.moc"
