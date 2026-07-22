#!/bin/bash
# WebGPU Shader Transformation Test
# This script verifies that the WebGPU shader transformation is working correctly

set -e

echo "=== NCNN WebGPU Shader Transformation Test ==="

# Test directory
TEST_DIR="/tmp/ncnn_webgpu_test"
mkdir -p "$TEST_DIR"

# Create a test shader with push constants
cat > "$TEST_DIR/test_shader.comp" << 'EOF'
#version 450

layout (constant_id = 0) const int w = 0;
layout (constant_id = 1) const int h = 0;

layout (binding = 0) buffer bottom_blob { float data[]; };

layout (push_constant) uniform parameter
{
    int dims;
    int w;
    int h;
    int c;
    int cstep;
} p;

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    
    if (gx >= psc(w) || gy >= psc(h))
        return;
        
    int gi = gy * psc(w) + gx;
    data[gi] = data[gi] * 2.0;
}
EOF

# Run the WebGPU transformation
echo "Running WebGPU shader transformation..."
cd "$(dirname "$0")"
cmake -DSHADER_SRC="$TEST_DIR/test_shader.comp" \
      -DSHADER_COMP_HEADER="$TEST_DIR/output.h" \
      -P "cmake/ncnn_generate_webgpu_shader_header.cmake"

# Check if transformation worked
if [[ -f "$TEST_DIR/output.h" ]]; then
    echo "✅ Shader transformation completed successfully"
    
    # Extract and display the transformed shader
    echo "=== Transformed Shader Content ==="
    
    # Read the hex data and convert back to text
    hex_data=$(grep -o '0x[0-9a-f][0-9a-f]' "$TEST_DIR/output.h" | tr -d '\n' | sed 's/0x//g')
    echo "$hex_data" | xxd -r -p
    
    echo -e "\n=== Verification ==="
    
    # Check for WebGPU transformations
    transformed_text=$(echo "$hex_data" | xxd -r -p)
    
    if echo "$transformed_text" | grep -q "struct parameter"; then
        echo "✅ Push constant struct conversion: PASSED"
    else
        echo "❌ Push constant struct conversion: FAILED"
        exit 1
    fi
    
    if echo "$transformed_text" | grep -q "layout (binding = 1) uniform parameter_blob"; then
        echo "✅ Uniform binding layout: PASSED"
    else
        echo "❌ Uniform binding layout: FAILED"
        exit 1
    fi
    
    if ! echo "$transformed_text" | grep -q "layout (push_constant)"; then
        echo "✅ Push constant removal: PASSED"
    else
        echo "❌ Push constant removal: FAILED"
        exit 1
    fi
    
    echo -e "\n🎉 All WebGPU shader transformations verified successfully!"
    
else
    echo "❌ Shader transformation failed - output file not created"
    exit 1
fi

# Cleanup
rm -rf "$TEST_DIR"

echo "=== Test completed successfully ==="