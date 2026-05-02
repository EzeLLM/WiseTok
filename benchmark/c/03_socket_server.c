/*
 * socket_server.c - POSIX socket server with epoll event loop
 *
 * Implements a non-blocking TCP server using epoll on Linux,
 * with signal handling and per-connection processing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

#define MAX_EVENTS 64
#define MAX_CONNECTIONS 1024
#define BUFFER_SIZE 4096
#define SERVER_PORT 8888

typedef struct {
  int fd;
  char buffer[BUFFER_SIZE];
  size_t buf_len;
} client_t;

static volatile sig_atomic_t shutdown_flag = 0;
static int epoll_fd = -1;
static client_t clients[MAX_CONNECTIONS];

/* Signal handler for graceful shutdown */
static void signal_handler(int sig)
{
  if (sig == SIGINT || sig == SIGTERM) {
    shutdown_flag = 1;
  }
}

/**
 * set_nonblocking - Configure a socket for non-blocking I/O
 * @fd: File descriptor to configure
 *
 * Sets the O_NONBLOCK flag on the socket.
 * Returns 0 on success, -1 on error.
 */
static int set_nonblocking(int fd)
{
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) {
    perror("fcntl(F_GETFL)");
    return -1;
  }

  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
    perror("fcntl(F_SETFL)");
    return -1;
  }

  return 0;
}

/**
 * add_to_epoll - Register a file descriptor with epoll
 * @fd: File descriptor to monitor
 * @events: Event mask (EPOLLIN, EPOLLOUT, etc.)
 *
 * Adds the file descriptor to the epoll instance.
 * Returns 0 on success, -1 on error.
 */
static int add_to_epoll(int fd, uint32_t events)
{
  struct epoll_event event;
  event.events = events;
  event.data.fd = fd;

  if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event) == -1) {
    perror("epoll_ctl(ADD)");
    return -1;
  }

  return 0;
}

/**
 * remove_from_epoll - Unregister a file descriptor from epoll
 * @fd: File descriptor to stop monitoring
 *
 * Removes the file descriptor from the epoll instance.
 */
static void remove_from_epoll(int fd)
{
  if (epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL) == -1) {
    perror("epoll_ctl(DEL)");
  }
}

/**
 * close_connection - Clean up and close a client connection
 * @client: Client structure to close
 */
static void close_connection(client_t *client)
{
  if (client->fd != -1) {
    remove_from_epoll(client->fd);
    if (close(client->fd) == -1) {
      perror("close");
    }
    printf("Closed connection from fd %d\n", client->fd);
    client->fd = -1;
    client->buf_len = 0;
  }
}

/**
 * handle_client_data - Process data from a connected client
 * @client: Client structure with data to process
 *
 * Reads from the socket and echoes data back.
 * Returns 0 on success, -1 on error or connection close.
 */
static int handle_client_data(client_t *client)
{
  ssize_t n;

  /* Read from socket */
  n = recv(client->fd, client->buffer + client->buf_len,
           BUFFER_SIZE - client->buf_len, 0);

  if (n == -1) {
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      perror("recv");
      return -1;
    }
    return 0;
  }

  if (n == 0) {
    /* Client closed connection */
    return -1;
  }

  client->buf_len += n;

  /* Echo back what we received */
  ssize_t sent = 0;
  while (sent < (ssize_t)client->buf_len) {
    ssize_t ret = send(client->fd, client->buffer + sent,
                       client->buf_len - sent, MSG_NOSIGNAL);
    if (ret == -1) {
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        perror("send");
        return -1;
      }
      break;
    }
    sent += ret;
  }

  /* Shift unacknowledged data to buffer start */
  if (sent > 0) {
    if (sent < (ssize_t)client->buf_len) {
      memmove(client->buffer, client->buffer + sent, client->buf_len - sent);
    }
    client->buf_len -= sent;
  }

  return 0;
}

/**
 * accept_new_connection - Accept an incoming connection
 * @listen_fd: Listening socket file descriptor
 *
 * Accepts a new client connection and adds it to the epoll set.
 * Returns 0 on success, -1 on error.
 */
static int accept_new_connection(int listen_fd)
{
  struct sockaddr_in client_addr;
  socklen_t addr_len = sizeof(client_addr);
  int client_fd;
  int i;

  client_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &addr_len);
  if (client_fd == -1) {
    perror("accept");
    return -1;
  }

  printf("Accepted connection from %s:%d\n",
         inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

  /* Find free slot for new client */
  for (i = 0; i < MAX_CONNECTIONS; i++) {
    if (clients[i].fd == -1) {
      clients[i].fd = client_fd;
      clients[i].buf_len = 0;
      break;
    }
  }

  if (i == MAX_CONNECTIONS) {
    fprintf(stderr, "Max connections reached, closing client\n");
    close(client_fd);
    goto error;
  }

  if (set_nonblocking(client_fd) == -1) {
    goto error;
  }

  if (add_to_epoll(client_fd, EPOLLIN | EPOLLHUP | EPOLLERR) == -1) {
    goto error;
  }

  return 0;

error:
  close_connection(&clients[i]);
  return -1;
}

/**
 * run_server - Main event loop
 * @listen_fd: Listening socket
 *
 * Runs the epoll-based event loop until shutdown_flag is set.
 */
static int run_server(int listen_fd)
{
  struct epoll_event events[MAX_EVENTS];
  int nfds;
  int i;

  while (!shutdown_flag) {
    nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, 1000);

    if (nfds == -1) {
      if (errno == EINTR) {
        continue;
      }
      perror("epoll_wait");
      return -1;
    }

    for (i = 0; i < nfds; i++) {
      int fd = events[i].data.fd;

      if (fd == listen_fd) {
        /* Incoming connection */
        if (events[i].events & EPOLLIN) {
          if (accept_new_connection(listen_fd) == -1) {
            continue;
          }
        }
      } else {
        /* Client connection */
        int j;
        for (j = 0; j < MAX_CONNECTIONS; j++) {
          if (clients[j].fd == fd) {
            break;
          }
        }

        if (j == MAX_CONNECTIONS) {
          continue;
        }

        if (events[i].events & EPOLLERR || events[i].events & EPOLLHUP) {
          close_connection(&clients[j]);
        } else if (events[i].events & EPOLLIN) {
          if (handle_client_data(&clients[j]) == -1) {
            close_connection(&clients[j]);
          }
        }
      }
    }
  }

  return 0;
}

/**
 * main - Initialize server and run event loop
 * @argc: Argument count
 * @argv: Argument vector
 *
 * Sets up TCP server socket, registers signal handlers, and runs event loop.
 */
int main(int argc, char *argv[])
{
  struct sockaddr_in server_addr;
  int listen_fd = -1;
  int i;

  /* Initialize clients array */
  for (i = 0; i < MAX_CONNECTIONS; i++) {
    clients[i].fd = -1;
  }

  /* Set up signal handlers */
  if (signal(SIGINT, signal_handler) == SIG_ERR ||
      signal(SIGTERM, signal_handler) == SIG_ERR) {
    perror("signal");
    goto error;
  }

  /* Create listening socket */
  listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd == -1) {
    perror("socket");
    goto error;
  }

  int opt = 1;
  if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
    perror("setsockopt");
    goto error;
  }

  /* Bind to port */
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  server_addr.sin_port = htons(SERVER_PORT);

  if (bind(listen_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
    perror("bind");
    goto error;
  }

  if (listen(listen_fd, SOMAXCONN) == -1) {
    perror("listen");
    goto error;
  }

  printf("Server listening on port %d\n", SERVER_PORT);

  if (set_nonblocking(listen_fd) == -1) {
    goto error;
  }

  /* Create epoll instance */
  epoll_fd = epoll_create1(0);
  if (epoll_fd == -1) {
    perror("epoll_create1");
    goto error;
  }

  if (add_to_epoll(listen_fd, EPOLLIN) == -1) {
    goto error;
  }

  /* Run event loop */
  if (run_server(listen_fd) == -1) {
    goto error;
  }

  printf("Shutting down gracefully...\n");

  /* Close all connections */
  for (i = 0; i < MAX_CONNECTIONS; i++) {
    close_connection(&clients[i]);
  }

  if (listen_fd != -1) {
    close(listen_fd);
  }

  if (epoll_fd != -1) {
    close(epoll_fd);
  }

  return 0;

error:
  fprintf(stderr, "Server error, shutting down\n");
  for (i = 0; i < MAX_CONNECTIONS; i++) {
    close_connection(&clients[i]);
  }
  if (listen_fd != -1) {
    close(listen_fd);
  }
  if (epoll_fd != -1) {
    close(epoll_fd);
  }
  return 1;
}
