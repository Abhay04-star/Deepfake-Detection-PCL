// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title DeepfakeHashRegistry
/// @notice Stores SHA-256 hashes (bytes32) of original media for authenticity verification.
contract DeepfakeHashRegistry {
    struct Record {
        address submitter;
        uint256 timestamp;
        bool exists;
    }

    mapping(bytes32 => Record) private records;

    event HashStored(bytes32 indexed sha256Hash, address indexed submitter, uint256 timestamp);

    /// @notice Store a SHA-256 hash if not already stored.
    function storeHash(bytes32 sha256Hash) external {
        require(sha256Hash != bytes32(0), "Invalid hash");
        require(!records[sha256Hash].exists, "Already stored");

        records[sha256Hash] = Record({submitter: msg.sender, timestamp: block.timestamp, exists: true});
        emit HashStored(sha256Hash, msg.sender, block.timestamp);
    }

    /// @notice Check whether a hash exists and retrieve metadata.
    function getRecord(bytes32 sha256Hash) external view returns (bool exists, address submitter, uint256 timestamp) {
        Record memory r = records[sha256Hash];
        return (r.exists, r.submitter, r.timestamp);
    }
}

