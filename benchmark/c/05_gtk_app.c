/*
 * gtk_app.c - GTK4 Desktop Application Example
 *
 * Demonstrates GTK4 patterns: GApplication, signal handling, GtkBuilder,
 * callbacks, and Cairo drawing primitives.
 */

#include <gtk/gtk.h>
#include <glib/gi18n.h>
#include <cairo.h>
#include <math.h>

typedef struct {
  GtkApplication *app;
  GtkWindow *window;
  GtkBox *main_box;
  GtkButton *btn_draw;
  GtkButton *btn_clear;
  GtkDrawingArea *canvas;
  cairo_t *cr;
  gboolean is_drawing;
} app_state_t;

static app_state_t app_state = { 0 };

/**
 * on_draw_event - Canvas draw callback
 * @area: Drawing area widget
 * @cr: Cairo context
 * @width: Canvas width
 * @height: Canvas height
 * @user_data: Application state
 *
 * Renders graphics to the canvas using Cairo.
 */
static void on_draw_event(GtkDrawingArea *area,
                         cairo_t *cr,
                         int width,
                         int height,
                         gpointer user_data)
{
  app_state_t *state = (app_state_t *)user_data;

  /* Clear canvas */
  cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_fill(cr);

  if (!state->is_drawing) {
    return;
  }

  /* Draw decorative pattern */
  cairo_set_source_rgb(cr, 0.2, 0.4, 0.8);
  cairo_set_line_width(cr, 2.0);

  /* Draw grid */
  for (int x = 0; x < width; x += 40) {
    cairo_move_to(cr, x, 0);
    cairo_line_to(cr, x, height);
    cairo_stroke(cr);
  }

  for (int y = 0; y < height; y += 40) {
    cairo_move_to(cr, 0, y);
    cairo_line_to(cr, width, y);
    cairo_stroke(cr);
  }

  /* Draw circles */
  cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
  cairo_set_line_width(cr, 3.0);

  double cx = width / 2.0;
  double cy = height / 2.0;

  for (int i = 0; i < 4; i++) {
    double angle = (i * M_PI / 2.0);
    double x = cx + 60 * cos(angle);
    double y = cy + 60 * sin(angle);
    cairo_arc(cr, x, y, 20, 0, 2 * M_PI);
    cairo_stroke(cr);
  }

  /* Draw central shape */
  cairo_set_source_rgb(cr, 0.0, 1.0, 0.0);
  cairo_arc(cr, cx, cy, 40, 0, 2 * M_PI);
  cairo_fill(cr);

  cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
  cairo_arc(cr, cx, cy, 40, 0, 2 * M_PI);
  cairo_stroke(cr);
}

/**
 * on_draw_button_clicked - Handler for Draw button
 * @button: Button widget
 * @user_data: Application state
 *
 * Toggles drawing state and queues canvas redraw.
 */
static void on_draw_button_clicked(GtkButton *button, gpointer user_data)
{
  app_state_t *state = (app_state_t *)user_data;

  state->is_drawing = TRUE;
  gtk_widget_queue_draw(GTK_WIDGET(state->canvas));
  gtk_button_set_label(button, _("Drawing..."));
}

/**
 * on_clear_button_clicked - Handler for Clear button
 * @button: Button widget
 * @user_data: Application state
 *
 * Clears the canvas and resets drawing state.
 */
static void on_clear_button_clicked(GtkButton *button, gpointer user_data)
{
  app_state_t *state = (app_state_t *)user_data;

  state->is_drawing = FALSE;
  gtk_widget_queue_draw(GTK_WIDGET(state->canvas));
  gtk_button_set_label(GTK_BUTTON(state->btn_draw), _("Draw"));
}

/**
 * on_activate - GApplication activation signal
 * @app: GApplication instance
 * @user_data: Application state
 *
 * Creates the main window and initializes UI.
 */
static void on_activate(GApplication *app, gpointer user_data)
{
  app_state_t *state = (app_state_t *)user_data;

  state->app = GTK_APPLICATION(app);

  /* Create main window */
  state->window = GTK_WINDOW(gtk_application_window_new(state->app));
  gtk_window_set_title(state->window, _("GTK Drawing Application"));
  gtk_window_set_default_size(state->window, 600, 400);

  /* Create main container */
  state->main_box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_VERTICAL, 10));
  gtk_widget_set_margin_top(GTK_WIDGET(state->main_box), 10);
  gtk_widget_set_margin_bottom(GTK_WIDGET(state->main_box), 10);
  gtk_widget_set_margin_start(GTK_WIDGET(state->main_box), 10);
  gtk_widget_set_margin_end(GTK_WIDGET(state->main_box), 10);

  gtk_window_set_child(state->window, GTK_WIDGET(state->main_box));

  /* Create button box */
  GtkBox *button_box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5));

  state->btn_draw = GTK_BUTTON(gtk_button_new_with_label(_("Draw")));
  state->btn_clear = GTK_BUTTON(gtk_button_new_with_label(_("Clear")));

  gtk_box_append(button_box, GTK_WIDGET(state->btn_draw));
  gtk_box_append(button_box, GTK_WIDGET(state->btn_clear));

  gtk_box_append(state->main_box, GTK_WIDGET(button_box));

  /* Create drawing area */
  state->canvas = GTK_DRAWING_AREA(gtk_drawing_area_new());
  gtk_widget_set_hexpand(GTK_WIDGET(state->canvas), TRUE);
  gtk_widget_set_vexpand(GTK_WIDGET(state->canvas), TRUE);

  gtk_drawing_area_set_draw_func(state->canvas, on_draw_event, state, NULL);

  gtk_box_append(state->main_box, GTK_WIDGET(state->canvas));

  /* Connect signals */
  g_signal_connect(state->btn_draw, "clicked",
                   G_CALLBACK(on_draw_button_clicked), state);
  g_signal_connect(state->btn_clear, "clicked",
                   G_CALLBACK(on_clear_button_clicked), state);

  /* Show window */
  gtk_window_present(state->window);
}

/**
 * main - Application entry point
 * @argc: Argument count
 * @argv: Argument vector
 *
 * Initializes and runs the GTK application.
 */
int main(int argc, char **argv)
{
  GApplication *app;
  int status;

  /* Bind text domain for i18n */
  bindtextdomain("gtk_app", "/usr/share/locale");
  bind_textdomain_codeset("gtk_app", "UTF-8");
  textdomain("gtk_app");

  /* Create application */
  app = g_object_new(GTK_TYPE_APPLICATION,
                    "application-id", "org.example.GTKApp",
                    "flags", G_APPLICATION_FLAGS_NONE,
                    NULL);

  /* Connect activation signal */
  g_signal_connect(app, "activate", G_CALLBACK(on_activate), &app_state);

  /* Run application */
  status = g_application_run(app, argc, argv);

  g_object_unref(app);

  return status;
}
