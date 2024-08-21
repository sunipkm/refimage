macro_rules! rotate {
    ($v0:ident <- $v1:ident) => {{
        std::mem::swap(&mut $v0, &mut $v1);
    }};
    ($v0:ident <- $v1:ident <- $v2:ident) => {{
        let rot = $v0;
        $v0 = $v1;
        $v1 = $v2;
        $v2 = rot;
    }};
    ($v0:ident <- $v1:ident <- $v2:ident <- $v3:ident <- $v4:ident <- $v5:ident <- $v6:ident) => {{
        let rot = $v0;
        $v0 = $v1;
        $v1 = $v2;
        $v2 = $v3;
        $v3 = $v4;
        $v4 = $v5;
        $v5 = $v6;
        $v6 = rot;
    }};
}
