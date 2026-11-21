/// models.dev v1, cut down to the models the v2 feed lacks (written by `xtask refresh-catalog`).
pub const API_JSON: &str = include_str!("../../catalog/.snapshot/api.json");
/// models.dev's v2 feed, cut down to the fields the catalog reads (written by `xtask refresh-catalog`).
pub const API_V2_JSON: &str = include_str!("../../catalog/.snapshot/api_v2.json");
