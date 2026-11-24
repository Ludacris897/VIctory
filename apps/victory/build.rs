// Exports `D3D12SDKVersion` and `D3D12SDKPath` from victory.exe: how a program
// opts into the D3D12 Agility SDK. Windows 10's own runtime refuses the
// directly indexed heaps translated shaders use (root signature flags 0xC00).
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo:rustc-link-arg-bin=victory=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bin=victory=/EXPORT:D3D12SDKPath,DATA");
    }
}
