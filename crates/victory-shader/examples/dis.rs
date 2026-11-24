use victory_shader::*;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let c = Container::parse(&bytes).unwrap();
    let dxil = c.part(FourCC::DXIL).unwrap();
    for p in native_programs(dxil.data).unwrap() {
        println!("stage {:?} wave64 {} vgprs {} io {:?}", p.stage, p.wave64, p.vgprs, p.io);
        println!("layout {:?}", p.layout);
        let code = p.patched_code();
        for i in gfx10::decode(&code).unwrap() { println!("{:04x} {}", i.offset, i.text(p.wave64)); }
    }
    let t = if std::env::var("NV").is_ok() { emit::Target { lanes_min: 32, lanes_max: 32 } } else { emit::Target::ANY };
    match emit::translate_for(&bytes, t) {
        Ok(h) => { println!("translated: dxil {} bytes\n{}", h.dxil.len(), emit::last_module_summary()); if let Ok(o) = std::env::var("OUT") { std::fs::write(o, &h.dxil).unwrap(); } }
        Err(e) => println!("translate error: {e}"),
    }
}
