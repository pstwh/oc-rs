use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Debug, Clone)]
pub enum OcrsFormat {
    Json,
    Text,
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct OcrsArgs {
    pub file_path: String,

    #[clap(value_enum, default_value_t=OcrsFormat::Json)]
    pub format: OcrsFormat,

    pub resize: bool,
}
