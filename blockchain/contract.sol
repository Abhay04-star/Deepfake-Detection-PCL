// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title DeepfakeHashRegistry (String hash version)
/// @notice Stores SHA-256 file hashes (as strings) to verify authenticity.
contract DeepfakeHashRegistry {
    // We cannot use `string` directly as a mapping key, so we store a deterministic key:
    // keccak256(utf8(hashString)).
    mapping(bytes32 => bool) private stored;

    /// @notice Store a SHA-256 hex digest string.
    /// @param hash SHA-256 hex string (64 hex chars recommended)
    function storeHash(string memory hash) external {
        bytes32 key = keccak256(abi.encodePacked(hash));
        require(!stored[key], "Already stored");
        stored[key] = true;
    }

    /// @notice Verify whether a SHA-256 hex digest string exists on-chain.
    /// @param hash SHA-256 hex string
    /// @return exists true if stored
    function verifyHash(string memory hash) external view returns (bool exists) {
        bytes32 key = keccak256(abi.encodePacked(hash));
        return stored[key];
    }
}

