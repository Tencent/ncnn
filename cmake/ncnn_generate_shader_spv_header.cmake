
function(ncnn_generate_shader_spv_header SHADER_SPV_HEADER SHADER_SPV_HEX_HEADERS SHADER_SRC)

    # fp32
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=vec2 -Dsfpvec4=vec4 -Dsfpvec8=mat2x4 -Dsfpmat4=mat4
             -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpvec8=mat2x4 -Dafpmat4=mat4
             "-D buffer_ld1(buf,i)=buf[i]"
             "-D buffer_st1(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}"
             "-D buffer_ld2(buf,i)=buf[i]"
             "-D buffer_st2(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=buf[i]"
             "-D buffer_st4(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={vec4 _v=sbuf[si]; buf[i4.r]=_v.r;buf[i4.g]=_v.g;buf[i4.b]=_v.b;buf[i4.a]=_v.a;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=mat2x4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=buf[i]"
             "-D buffer_st8(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}"
             "-D sfp2afpmat4(v)=v"
             "-D afp2sfpmat4(v)=v"
             "-D psc(x)=(x==0?p.x:x)"
             -V -s -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # fp16 packed
    set(SHADER_fp16p_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16p")

    set(SHADER_fp16p_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16p_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16p_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=uint -Dsfpvec4=uvec2 -Dsfpvec8=uvec4
             -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4  -Dafpvec8=mat2x4 -Dafpmat4=mat4
             "-D buffer_ld1(buf,i)=buf[i]"
             "-D buffer_st1(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=uvec2(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])));}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=uvec4(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])),packHalf2x16(vec2(sbuf[sii4.r],sbuf[sii4.g])),packHalf2x16(vec2(sbuf[sii4.b],sbuf[sii4.a])));}"
             "-D buffer_ld2(buf,i)=unpackHalf2x16(buf[i])"
             "-D buffer_st2(buf,i,v)={buf[i]=packHalf2x16(v)}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y))"
             "-D buffer_st4(buf,i,v)={buf[i]=uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba));}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=mat2x4(vec4(unpackHalf2x16(buf[i].r),unpackHalf2x16(buf[i].g)),vec4(unpackHalf2x16(buf[i].b),unpackHalf2x16(buf[i].a)))"
             "-D buffer_st8(buf,i,v)={buf[i]=uvec4(uvec2(packHalf2x16(v[0].rg),packHalf2x16(v[0].ba)),uvec2(packHalf2x16(v[1].rg),packHalf2x16(v[1].ba)));}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}"
             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_fp16_packed=1
             -V -s -x -o ${SHADER_fp16p_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_fp16p_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16p_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # fp16 packed + fp16 arithmetic
    set(SHADER_fp16pa_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16pa")

    set(SHADER_fp16pa_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16pa_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16pa_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=uint -Dsfpvec4=uvec2 -Dsfpvec8=uvec4
             -Dafp=float16_t -Dafpvec2=f16vec2 -Dafpvec4=f16vec4  -Dafpvec8=f16mat2x4 -Dafpmat4=f16mat4
             "-D buffer_ld1(buf,i)=float16_t(buf[i])"
             "-D buffer_st1(buf,i,v)={buf[i]=float(v);}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=uvec2(packHalf2x16(vec2(f16vec2(sbuf[si4.r],sbuf[si4.g]))),packHalf2x16(vec2(f16vec2(sbuf[si4.b],sbuf[si4.a]))));}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=uvec4(packHalf2x16(vec2(f16vec2(sbuf[si4.r],sbuf[si4.g]))),packHalf2x16(vec2(f16vec2(sbuf[si4.b],sbuf[si4.a]))),packHalf2x16(vec2(f16vec2(sbuf[sii4.r],sbuf[sii4.g]))),packHalf2x16(vec2(f16vec2(sbuf[sii4.b],sbuf[sii4.a]))));}"
             "-D buffer_ld2(buf,i)=f16vec2(unpackHalf2x16(buf[i]))"
             "-D buffer_st2(buf,i,v)={buf[i]=packHalf2x16(vec2(v))}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=f16vec4(vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y)))"
             "-D buffer_st4(buf,i,v)={buf[i]=uvec2(packHalf2x16(vec2(v.rg)),packHalf2x16(vec2(v.ba)));}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=f16mat2x4(f16vec4(vec4(unpackHalf2x16(buf[i].r),unpackHalf2x16(buf[i].g))),f16vec4(vec4(unpackHalf2x16(buf[i].b),unpackHalf2x16(buf[i].a))))"
             "-D buffer_st8(buf,i,v)={buf[i]=uvec4(uvec2(packHalf2x16(vec2(v[0].rg)),packHalf2x16(vec2(v[0].ba))),uvec2(packHalf2x16(vec2(v[1].rg)),packHalf2x16(vec2(v[1].ba))));}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}"
             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_fp16_packed=1 -DNCNN_fp16_arithmetic=1
             -V -s -x -o ${SHADER_fp16pa_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_fp16pa_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16pa_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # fp16 storage
    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4
             -Dafp=float     -Dafpvec2=vec2    -Dafpvec4=vec4    -Dafpvec8=mat2x4 -Dafpmat4=mat4
             "-D buffer_ld1(buf,i)=float(buf[i])"
             "-D buffer_st1(buf,i,v)={buf[i]=float16_t(v);}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i].r=sbuf[si4.r];buf[i].g=sbuf[si4.g];buf[i].b=sbuf[si4.b];buf[i].a=sbuf[si4.a];}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i].abcd.r=sbuf[si4.r];buf[i].abcd.g=sbuf[si4.g];buf[i].abcd.b=sbuf[si4.b];buf[i].abcd.a=sbuf[si4.a];buf[i].efgh.r=sbuf[sii4.r];buf[i].efgh.g=sbuf[sii4.g];buf[i].efgh.b=sbuf[sii4.b];buf[i].efgh.a=sbuf[sii4.a];}"
             "-D buffer_ld2(buf,i)=vec2(buf[i])"
             "-D buffer_st2(buf,i,v)={buf[i]=f16vec2(v);}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=vec4(buf[i])"
             "-D buffer_st4(buf,i,v)={buf[i]=f16vec4(v);}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i].abcd=sbuf[si2.r];buf[i].efgh=sbuf[si2.g];}"
             "-D buffer_ld8(buf,i)=mat2x4(vec4(buf[i].abcd),vec4(buf[i].efgh))"
             "-D buffer_st8(buf,i,v)={buf[i].abcd=f16vec4(v[0]);buf[i].efgh=f16vec4(v[1]);}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i].abcd=sbuf[si].abcd;buf[i].efgh=sbuf[si].efgh;}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={buf[i4.r]=sbuf[si].abcd.r;buf[i4.g]=sbuf[si].abcd.g;buf[i4.b]=sbuf[si].abcd.b;buf[i4.a]=sbuf[si].abcd.a; buf[ii4.r]=sbuf[si].efgh.r;buf[ii4.g]=sbuf[si].efgh.g;buf[ii4.b]=sbuf[si].efgh.b;buf[ii4.a]=sbuf[si].efgh.a;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={buf[i2.r]=sbuf[si].abcd;buf[i2.g]=sbuf[si].efgh;}"
             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_fp16_storage=1
             -V -s -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # fp16 storage + fp16 arithmetic
    set(SHADER_fp16sa_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16sa")

    set(SHADER_fp16sa_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16sa_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16sa_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4 -Dsfpvec8=f16mat2x4 -Dsfpmat4=f16mat4
             -Dafp=float16_t -Dafpvec2=f16vec2 -Dafpvec4=f16vec4 -Dafpvec8=f16mat2x4 -Dafpmat4=f16mat4
             "-D buffer_ld1(buf,i)=buf[i]"
             "-D buffer_st1(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=f16vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=f16mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}"
             "-D buffer_ld2(buf,i)=buf[i]"
             "-D buffer_st2(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=buf[i]"
             "-D buffer_st4(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=f16mat2x4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=buf[i]"
             "-D buffer_st8(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={f16mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={f16mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}"
             "-D sfp2afpmat4(v)=v"
             "-D afp2sfpmat4(v)=v"
             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_fp16_storage=1 -DNCNN_fp16_arithmetic=1
             -V -s -x -o ${SHADER_fp16sa_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_fp16sa_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16sa_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # image + fp32
    set(SHADER_image_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_image")

    set(SHADER_image_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_image_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_image_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=vec2 -Dsfpvec4=vec4 -Dsfpvec8=mat2x4 -Dsfpmat4=mat4
             -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpvec8=mat2x4 -Dafpmat4=mat4

             -Dimfmtc1=r32f -Dimfmtc4=rgba32f
             -Dunfp=highp

             "-D image1d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image2d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image3d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image1d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image2d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image3d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image1d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image2d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image3d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image1d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image2d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image3d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image1d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld8(tex,p)=mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"
             "-D image2d_ld8(tex,p)=mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"
             "-D image3d_ld8(tex,p)=mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"
             "-D image1d_st8(img,p,v)={imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"
             "-D image2d_st8(img,p,v)={imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"
             "-D image3d_st8(img,p,v)={imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"
             "-D image1d_cp8(img,p,tex,sp)={imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"
             "-D image2d_cp8(img,p,tex,sp)={imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"
             "-D image3d_cp8(img,p,tex,sp)={imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"

             "-D buffer_ld1(buf,i)=buf[i]"
             "-D buffer_st1(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}"
             "-D buffer_ld2(buf,i)=buf[i]"
             "-D buffer_st2(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=buf[i]"
             "-D buffer_st4(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={vec4 _v=sbuf[si]; buf[i4.r]=_v.r;buf[i4.g]=_v.g;buf[i4.b]=_v.b;buf[i4.a]=_v.a;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=mat2x4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=buf[i]"
             "-D buffer_st8(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}"

             "-D sfp2afpmat4(v)=v"
             "-D afp2sfpmat4(v)=v"
             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_image_shader=1
             -V -s -x -o ${SHADER_image_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_image_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_image_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # image + fp16p
    set(SHADER_image_fp16p_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_image_fp16p")

    set(SHADER_image_fp16p_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_image_fp16p_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_image_fp16p_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=uint -Dsfpvec4=uvec2 -Dsfpvec8=uvec4
             -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4  -Dafpvec8=mat2x4 -Dafpmat4=mat4

             -Dimfmtc1=r32f -Dimfmtc4=rgba16f
             -Dunfp=mediump

             "-D image1d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image2d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image3d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image1d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image2d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image3d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image1d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image2d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image3d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image1d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image2d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image3d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image1d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld8(tex,p)=mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"
             "-D image2d_ld8(tex,p)=mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"
             "-D image3d_ld8(tex,p)=mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"
             "-D image1d_st8(img,p,v)={imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"
             "-D image2d_st8(img,p,v)={imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"
             "-D image3d_st8(img,p,v)={imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"
             "-D image1d_cp8(img,p,tex,sp)={imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"
             "-D image2d_cp8(img,p,tex,sp)={imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"
             "-D image3d_cp8(img,p,tex,sp)={imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"

             "-D buffer_ld1(buf,i)=buf[i]"
             "-D buffer_st1(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=uvec2(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])));}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=uvec4(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])),packHalf2x16(vec2(sbuf[sii4.r],sbuf[sii4.g])),packHalf2x16(vec2(sbuf[sii4.b],sbuf[sii4.a])));}"
             "-D buffer_ld2(buf,i)=unpackHalf2x16(buf[i])"
             "-D buffer_st2(buf,i,v)={buf[i]=packHalf2x16(v)}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y))"
             "-D buffer_st4(buf,i,v)={buf[i]=uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba));}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=mat2x4(vec4(unpackHalf2x16(buf[i].r),unpackHalf2x16(buf[i].g)),vec4(unpackHalf2x16(buf[i].b),unpackHalf2x16(buf[i].a)))"
             "-D buffer_st8(buf,i,v)={buf[i]=uvec4(uvec2(packHalf2x16(v[0].rg),packHalf2x16(v[0].ba)),uvec2(packHalf2x16(v[1].rg),packHalf2x16(v[1].ba)));}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}"

             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_image_shader=1 -DNCNN_fp16_packed=1
             -V -s -x -o ${SHADER_image_fp16p_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_image_fp16p_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_image_fp16p_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # image + fp16p + fp16a
    set(SHADER_image_fp16pa_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_image_fp16pa")

    set(SHADER_image_fp16pa_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_image_fp16pa_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_image_fp16pa_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=uint -Dsfpvec4=uvec2 -Dsfpvec8=uvec4
             -Dafp=float16_t -Dafpvec2=f16vec2 -Dafpvec4=f16vec4  -Dafpvec8=f16mat2x4 -Dafpmat4=f16mat4

             -Dimfmtc1=r32f -Dimfmtc4=rgba16f
             -Dunfp=mediump

             "-D image1d_ld1(tex,p)=float16_t(texelFetch(tex,p,0).r)"
             "-D image2d_ld1(tex,p)=float16_t(texelFetch(tex,p,0).r)"
             "-D image3d_ld1(tex,p)=float16_t(texelFetch(tex,p,0).r)"
             "-D image1d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image2d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image3d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image1d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld4(tex,p)=f16vec4(texelFetch(tex,p,0))"
             "-D image2d_ld4(tex,p)=f16vec4(texelFetch(tex,p,0))"
             "-D image3d_ld4(tex,p)=f16vec4(texelFetch(tex,p,0))"
             "-D image1d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image2d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image3d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image1d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld8(tex,p)=f16mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"
             "-D image2d_ld8(tex,p)=f16mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"
             "-D image3d_ld8(tex,p)=f16mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"
             "-D image1d_st8(img,p,v)={imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"
             "-D image2d_st8(img,p,v)={imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"
             "-D image3d_st8(img,p,v)={imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"
             "-D image1d_cp8(img,p,tex,sp)={imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"
             "-D image2d_cp8(img,p,tex,sp)={imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"
             "-D image3d_cp8(img,p,tex,sp)={imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"

             "-D buffer_ld1(buf,i)=float16_t(buf[i])"
             "-D buffer_st1(buf,i,v)={buf[i]=float(v);}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=uvec2(packHalf2x16(vec2(f16vec2(sbuf[si4.r],sbuf[si4.g]))),packHalf2x16(vec2(f16vec2(sbuf[si4.b],sbuf[si4.a]))));}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=uvec4(packHalf2x16(vec2(f16vec2(sbuf[si4.r],sbuf[si4.g]))),packHalf2x16(vec2(f16vec2(sbuf[si4.b],sbuf[si4.a]))),packHalf2x16(vec2(f16vec2(sbuf[sii4.r],sbuf[sii4.g]))),packHalf2x16(vec2(f16vec2(sbuf[sii4.b],sbuf[sii4.a]))));}"
             "-D buffer_ld2(buf,i)=f16vec2(unpackHalf2x16(buf[i]))"
             "-D buffer_st2(buf,i,v)={buf[i]=packHalf2x16(vec2(v))}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=f16vec4(vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y)))"
             "-D buffer_st4(buf,i,v)={buf[i]=uvec2(packHalf2x16(vec2(v.rg)),packHalf2x16(vec2(v.ba)));}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=f16mat2x4(f16vec4(vec4(unpackHalf2x16(buf[i].r),unpackHalf2x16(buf[i].g))),f16vec4(vec4(unpackHalf2x16(buf[i].b),unpackHalf2x16(buf[i].a))))"
             "-D buffer_st8(buf,i,v)={buf[i]=uvec4(uvec2(packHalf2x16(vec2(v[0].rg)),packHalf2x16(vec2(v[0].ba))),uvec2(packHalf2x16(vec2(v[1].rg)),packHalf2x16(vec2(v[1].ba))));}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}"

             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_image_shader=1 -DNCNN_fp16_packed=1 -DNCNN_fp16_arithmetic=1
             -V -s -x -o ${SHADER_image_fp16pa_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_image_fp16pa_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_image_fp16pa_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # image + fp16s
    set(SHADER_image_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_image_fp16s")

    set(SHADER_image_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_image_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_image_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4
             -Dafp=float     -Dafpvec2=vec2    -Dafpvec4=vec4    -Dafpvec8=mat2x4 -Dafpmat4=mat4

             -Dimfmtc1=r16f -Dimfmtc4=rgba16f
             -Dunfp=mediump

             "-D image1d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image2d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image3d_ld1(tex,p)=texelFetch(tex,p,0).r"
             "-D image1d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image2d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image3d_st1(img,p,v)={vec4 _v;_v.r=v;imageStore(img,p,_v);}"
             "-D image1d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image2d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image3d_ld4(tex,p)=texelFetch(tex,p,0)"
             "-D image1d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image2d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image3d_st4(img,p,v)={imageStore(img,p,v);}"
             "-D image1d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld8(tex,p)=mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"
             "-D image2d_ld8(tex,p)=mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"
             "-D image3d_ld8(tex,p)=mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"
             "-D image1d_st8(img,p,v)={imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"
             "-D image2d_st8(img,p,v)={imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"
             "-D image3d_st8(img,p,v)={imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"
             "-D image1d_cp8(img,p,tex,sp)={imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"
             "-D image2d_cp8(img,p,tex,sp)={imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"
             "-D image3d_cp8(img,p,tex,sp)={imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"

             "-D buffer_ld1(buf,i)=float(buf[i])"
             "-D buffer_st1(buf,i,v)={buf[i]=float16_t(v);}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i].r=sbuf[si4.r];buf[i].g=sbuf[si4.g];buf[i].b=sbuf[si4.b];buf[i].a=sbuf[si4.a];}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i].abcd.r=sbuf[si4.r];buf[i].abcd.g=sbuf[si4.g];buf[i].abcd.b=sbuf[si4.b];buf[i].abcd.a=sbuf[si4.a];buf[i].efgh.r=sbuf[sii4.r];buf[i].efgh.g=sbuf[sii4.g];buf[i].efgh.b=sbuf[sii4.b];buf[i].efgh.a=sbuf[sii4.a];}"
             "-D buffer_ld2(buf,i)=vec2(buf[i])"
             "-D buffer_st2(buf,i,v)={buf[i]=f16vec2(v);}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=vec4(buf[i])"
             "-D buffer_st4(buf,i,v)={buf[i]=f16vec4(v);}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i].abcd=sbuf[si2.r];buf[i].efgh=sbuf[si2.g];}"
             "-D buffer_ld8(buf,i)=mat2x4(vec4(buf[i].abcd),vec4(buf[i].efgh))"
             "-D buffer_st8(buf,i,v)={buf[i].abcd=f16vec4(v[0]);buf[i].efgh=f16vec4(v[1]);}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i].abcd=sbuf[si].abcd;buf[i].efgh=sbuf[si].efgh;}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={buf[i4.r]=sbuf[si].abcd.r;buf[i4.g]=sbuf[si].abcd.g;buf[i4.b]=sbuf[si].abcd.b;buf[i4.a]=sbuf[si].abcd.a; buf[ii4.r]=sbuf[si].efgh.r;buf[ii4.g]=sbuf[si].efgh.g;buf[ii4.b]=sbuf[si].efgh.b;buf[ii4.a]=sbuf[si].efgh.a;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={buf[i2.r]=sbuf[si].abcd;buf[i2.g]=sbuf[si].efgh;}"

             "-D sfp2afpmat4(v)=v"
             "-D afp2sfpmat4(v)=v"
             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_image_shader=1 -DNCNN_fp16_storage=1
             -V -s -x -o ${SHADER_image_fp16s_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_image_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_image_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # image + fp16s + fp16a
    set(SHADER_image_fp16sa_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_image_fp16sa")

    set(SHADER_image_fp16sa_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_image_fp16sa_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_image_fp16sa_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4 -Dsfpvec8=f16mat2x4 -Dsfpmat4=f16mat4
             -Dafp=float16_t -Dafpvec2=f16vec2 -Dafpvec4=f16vec4 -Dafpvec8=f16mat2x4 -Dafpmat4=f16mat4

             -Dimfmtc1=r16f -Dimfmtc4=rgba16f
             -Dunfp=mediump

             "-D image1d_ld1(tex,p)=float16_t(texelFetch(tex,p,0).r)"
             "-D image2d_ld1(tex,p)=float16_t(texelFetch(tex,p,0).r)"
             "-D image3d_ld1(tex,p)=float16_t(texelFetch(tex,p,0).r)"
             "-D image1d_st1(img,p,v)={f16vec4 _v;_v.r=float16_t(v);imageStore(img,p,_v);}"
             "-D image2d_st1(img,p,v)={f16vec4 _v;_v.r=float16_t(v);imageStore(img,p,_v);}"
             "-D image3d_st1(img,p,v)={f16vec4 _v;_v.r=float16_t(v);imageStore(img,p,_v);}"
             "-D image1d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp1(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld4(tex,p)=f16vec4(texelFetch(tex,p,0))"
             "-D image2d_ld4(tex,p)=f16vec4(texelFetch(tex,p,0))"
             "-D image3d_ld4(tex,p)=f16vec4(texelFetch(tex,p,0))"
             "-D image1d_st4(img,p,v)={imageStore(img,p,vec4(v));}"
             "-D image2d_st4(img,p,v)={imageStore(img,p,vec4(v));}"
             "-D image3d_st4(img,p,v)={imageStore(img,p,vec4(v));}"
             "-D image1d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image2d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"
             "-D image3d_cp4(img,p,tex,sp)={imageStore(img,p,texelFetch(tex,sp,0));}"

             "-D image1d_ld8(tex,p)=f16mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"
             "-D image2d_ld8(tex,p)=f16mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"
             "-D image3d_ld8(tex,p)=f16mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"
             "-D image1d_st8(img,p,v)={imageStore(img,(p)*2,vec4(v[0]));imageStore(img,(p)*2+1,vec4(v[1]));}"
             "-D image2d_st8(img,p,v)={imageStore(img,ivec2(p.x*2,p.y),vec4(v[0]));imageStore(img,ivec2(p.x*2+1,p.y),vec4(v[1]));}"
             "-D image3d_st8(img,p,v)={imageStore(img,ivec3(p.x*2,p.y,p.z),vec4(v[0]));imageStore(img,ivec3(p.x*2+1,p.y,p.z),vec4(v[1]));}"
             "-D image1d_cp8(img,p,tex,sp)={imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"
             "-D image2d_cp8(img,p,tex,sp)={imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"
             "-D image3d_cp8(img,p,tex,sp)={imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"

             "-D buffer_ld1(buf,i)=buf[i]"
             "-D buffer_st1(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp1(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp1to4(buf,i,sbuf,si4)={buf[i]=f16vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}"
             "-D buffer_cp1to8(buf,i,sbuf,si4,sii4)={buf[i]=f16mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}"
             "-D buffer_ld2(buf,i)=buf[i]"
             "-D buffer_st2(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp2(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_ld4(buf,i)=buf[i]"
             "-D buffer_st4(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp4(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp4to1(buf,i4,sbuf,si)={buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}"
             "-D buffer_cp4to8(buf,i,sbuf,si2)={buf[i]=f16mat2x4(sbuf[si2.r],sbuf[si2.g]);}"
             "-D buffer_ld8(buf,i)=buf[i]"
             "-D buffer_st8(buf,i,v)={buf[i]=v;}"
             "-D buffer_cp8(buf,i,sbuf,si)={buf[i]=sbuf[si];}"
             "-D buffer_cp8to1(buf,i4,ii4,sbuf,si)={f16mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}"
             "-D buffer_cp8to4(buf,i2,sbuf,si)={f16mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}"
             "-D sfp2afpmat4(v)=v"
             "-D afp2sfpmat4(v)=v"

             "-D psc(x)=(x==0?p.x:x)"
             -DNCNN_image_shader=1 -DNCNN_fp16_storage=1 -DNCNN_fp16_arithmetic=1
             -V -s -x -o ${SHADER_image_fp16sa_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_image_fp16sa_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_image_fp16sa_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    set(LOCAL_SHADER_SPV_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.h)

    file(WRITE ${LOCAL_SHADER_SPV_HEADER}
        "static const uint32_t ${SHADER_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_fp16p_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16p_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_fp16pa_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16pa_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_fp16s_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_fp16sa_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16sa_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_image_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_image_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_image_fp16p_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_image_fp16p_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_image_fp16pa_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_image_fp16pa_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_image_fp16s_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_image_fp16s_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_image_fp16sa_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_image_fp16sa_SRC_NAME_WE}.spv.hex.h\"\n};\n"
    )

    set_source_files_properties(${LOCAL_SHADER_SPV_HEADER} PROPERTIES GENERATED TRUE)

    set(LOCAL_SHADER_SPV_HEX_HEADERS
        ${SHADER_SPV_HEX_FILE}
        ${SHADER_fp16p_SPV_HEX_FILE}
        ${SHADER_fp16pa_SPV_HEX_FILE}
        ${SHADER_fp16s_SPV_HEX_FILE}
        ${SHADER_fp16sa_SPV_HEX_FILE}
        ${SHADER_image_SPV_HEX_FILE}
        ${SHADER_image_fp16p_SPV_HEX_FILE}
        ${SHADER_image_fp16pa_SPV_HEX_FILE}
        ${SHADER_image_fp16s_SPV_HEX_FILE}
        ${SHADER_image_fp16sa_SPV_HEX_FILE}
    )

    set(${SHADER_SPV_HEADER} ${LOCAL_SHADER_SPV_HEADER} PARENT_SCOPE)
    set(${SHADER_SPV_HEX_HEADERS} ${LOCAL_SHADER_SPV_HEX_HEADERS} PARENT_SCOPE)

endfunction()
