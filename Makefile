
# Binary name (matches Cargo.toml -> [package].name)
BIN_NAME := compressed_inverted_index

# Index file generated during example/demo run
INDEX_FILE := example.idx

# ============================================================
# Core Targets
# ============================================================

# Build in debug mode
build:
	cargo build

# Build optimized release binary
release:
	cargo build --release

# Run with release optimizations
run:
	cargo run --release

# Run unit tests
test:
	cargo test

# Remove compiled artifacts and generated index file
clean:
	cargo clean
	@if [ -f $(INDEX_FILE) ]; then \
		echo "Removing $(INDEX_FILE)"; \
		rm -f $(INDEX_FILE); \
	fi

# Format all Rust source files
fmt:
	cargo fmt --all

# Run Clippy for linting (static analysis)
lint:
	cargo clippy --all-targets --all-features -- -D warnings

# Rebuild from scratch (clean + build)
rebuild: clean build

# Run the example (build + run)
example: release
	@echo "Running compressed inverted index example..."
	@cargo run --release

# ============================================================
# Help
# ============================================================

help:
	@echo "Available make targets:"
	@echo "  make build       - Build in debug mode"
	@echo "  make release     - Build optimized release binary"
	@echo "  make run         - Run in release mode"
	@echo "  make test        - Run unit tests"
	@echo "  make fmt         - Format code with rustfmt"
	@echo "  make lint        - Run Clippy for linting"
	@echo "  make clean       - Remove build artifacts and example.idx"
	@echo "  make rebuild     - Clean and rebuild everything"
	@echo "  make example     - Build and run the example demo"

.PHONY: build release run test clean fmt lint rebuild example help
