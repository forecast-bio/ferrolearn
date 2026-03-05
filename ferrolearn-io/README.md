# ferrolearn-io

Serialization and I/O utilities for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Format

Models are stored in a binary envelope format:
- **MessagePack** serialization via `rmp-serde`
- **CRC32** integrity checksum
- **Magic bytes** (`FLRN`) and schema version for forward compatibility

JSON export is also supported for interoperability.

## Functions

| Function | Description |
|----------|-------------|
| `save_model` | Serialize a model to a `.flrn` file |
| `load_model` | Load and validate a model from a `.flrn` file |
| `save_model_bytes` | Serialize to an in-memory `Vec<u8>` |
| `load_model_bytes` | Deserialize from bytes |
| `save_model_json` | Export as pretty-printed JSON |
| `load_model_json` | Load from a JSON file |

## Example

```rust
use ferrolearn_io::{save_model, load_model};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct MyModel {
    weights: Vec<f64>,
    bias: f64,
}

let model = MyModel { weights: vec![1.0, 2.0, 3.0], bias: 0.5 };

// Save and load (binary envelope)
save_model(&model, "model.flrn").unwrap();
let loaded: MyModel = load_model("model.flrn").unwrap();
assert_eq!(model, loaded);

// Or use JSON for inspection
ferrolearn_io::save_model_json(&model, "model.json").unwrap();
```

Any type implementing `serde::Serialize` and `serde::Deserialize` can be saved and loaded.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
