#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/status.h>

namespace pb {
    struct Request {
        std::string query;
        int limit = 10;
    };

    struct Response {
        std::string result;
        int status_code = 0;
        std::string error_message;
    };

    struct StreamMessage {
        std::string chunk;
        int sequence = 0;
    };
}

class SearchServiceImpl {
public:
    grpc::Status Search(
        grpc::ServerContext* context,
        const pb::Request* request,
        pb::Response* response) {

        if (request->query.empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Query cannot be empty");
        }

        if (request->limit < 1 || request->limit > 1000) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Limit must be between 1 and 1000");
        }

        std::string result = "Search results for: " + request->query;
        response->set_result(result);
        response->set_status_code(200);

        return grpc::Status::OK;
    }

    grpc::Status SearchStream(
        grpc::ServerContext* context,
        const pb::Request* request,
        grpc::ServerWriter<pb::StreamMessage>* writer) {

        if (request->query.empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Query cannot be empty");
        }

        std::string query = request->query;
        int total_chunks = 5;

        for (int i = 0; i < total_chunks; ++i) {
            if (context->IsCancelled()) {
                return grpc::Status(grpc::StatusCode::CANCELLED, "Stream cancelled by client");
            }

            pb::StreamMessage msg;
            msg.set_chunk("Result chunk " + std::to_string(i) + " for query: " + query);
            msg.set_sequence(i);

            if (!writer->Write(msg)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to write to stream");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        return grpc::Status::OK;
    }

    grpc::Status UploadStream(
        grpc::ServerContext* context,
        grpc::ServerReader<pb::StreamMessage>* reader,
        pb::Response* response) {

        pb::StreamMessage msg;
        int total_received = 0;
        std::string aggregated_data;

        while (reader->Read(&msg)) {
            total_received++;
            aggregated_data += msg.chunk();
            std::cout << "Received chunk " << msg.sequence() << ": " << msg.chunk() << std::endl;
        }

        response->set_result("Received " + std::to_string(total_received) + " chunks");
        response->set_status_code(200);

        return grpc::Status::OK;
    }

    grpc::Status BidirectionalStream(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<pb::StreamMessage, pb::StreamMessage>* stream) {

        pb::StreamMessage request_msg;

        while (stream->Read(&request_msg)) {
            if (context->IsCancelled()) {
                break;
            }

            pb::StreamMessage response_msg;
            response_msg.set_chunk("Echo: " + request_msg.chunk());
            response_msg.set_sequence(request_msg.sequence() + 1000);

            if (!stream->Write(response_msg)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to write response");
            }
        }

        return grpc::Status::OK;
    }
};

class SearchServer {
private:
    std::unique_ptr<grpc::Server> server;

public:
    void start(const std::string& address = "0.0.0.0:50051") {
        SearchServiceImpl service;

        grpc::ServerBuilder builder;
        builder.AddListeningPort(address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);

        std::cout << "Building server..." << std::endl;
        server = builder.BuildAndStart();

        if (server == nullptr) {
            std::cerr << "Failed to build server" << std::endl;
            return;
        }

        std::cout << "Server listening on " << address << std::endl;

        server->Wait();
    }

    void shutdown() {
        if (server) {
            server->Shutdown();
        }
    }
};

class AsyncSearchServer {
private:
    std::unique_ptr<grpc::ServerCompletionQueue> cq;
    std::unique_ptr<grpc::Server> server;

public:
    void start(const std::string& address = "0.0.0.0:50052") {
        SearchServiceImpl service;
        cq = std::make_unique<grpc::ServerCompletionQueue>();

        grpc::ServerBuilder builder;
        builder.AddListeningPort(address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        cq = builder.AddCompletionQueue();

        server = builder.BuildAndStart();

        if (server == nullptr) {
            std::cerr << "Failed to build async server" << std::endl;
            return;
        }

        std::cout << "Async server listening on " << address << std::endl;

        void* tag;
        bool ok;
        while (cq->Next(&tag, &ok)) {
            if (!ok) continue;
        }
    }

    void shutdown() {
        cq->Shutdown();
        if (server) {
            server->Shutdown();
        }
    }
};

int main(int argc, char** argv) {
    std::cout << "Starting gRPC Search Service" << std::endl;

    SearchServer sync_server;
    std::thread sync_thread([&sync_server]() {
        sync_server.start("0.0.0.0:50051");
    });

    std::cout << "Synchronous server running on port 50051" << std::endl;

    sync_thread.join();

    return 0;
}
