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
        for i in gfx10::decode(&code).unwrap() { println!("{}", i.text(p.wave64)); }
    }
    match emit::translate_for(&bytes, emit::Target::ANY) {
        Ok(h) => println!("translated: dxil {} bytes", h.dxil.len()),
        Err(e) => println!("translate error: {e}"),
    }
}
