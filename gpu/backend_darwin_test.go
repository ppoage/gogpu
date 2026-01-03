//go:build darwin

package gpu_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/gogpu/gogpu/gpu"
	"github.com/gogpu/gogpu/gpu/backend/native"
	"github.com/gogpu/gogpu/gpu/types"
	"github.com/gogpu/gogpu/internal/platform/darwin"
)

const backendTestWGSL = `
struct VertexInput {
	@location(0) position: vec2<f32>,
	@location(1) color: vec3<f32>,
}

struct VSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uTint: vec4<f32>;
@group(0) @binding(1) var uSampler: sampler;
@group(0) @binding(2) var uTexture: texture_2d<f32>;

@vertex
fn vs_main(input: VertexInput) -> VSOut {
	var out: VSOut;
	out.position = vec4<f32>(input.position, 0.0, 1.0);
	out.color = input.color;
	return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
	let texColor = textureSample(uTexture, uSampler, vec2<f32>(0.5, 0.5));
	return vec4<f32>(input.color, 1.0) * uTint * texColor;
}
`

func TestNativeBackendInterfaceDarwin(t *testing.T) {
	if err := runOnMainThread(func() error {
		backend := native.New()
		if backend == nil {
			return fmt.Errorf("native.New returned nil")
		}

		gpu.SetBackend(backend)
		defer gpu.SetBackend(nil)
		if got := gpu.GetBackend(); got != backend {
			return fmt.Errorf("GetBackend = %T, want %T", got, backend)
		}

		if err := backend.Init(); err != nil {
			return fmt.Errorf("Init failed: %w", err)
		}
		defer backend.Destroy()

		layer, err := newMetalLayer()
		if err != nil {
			return err
		}
		defer layer.Release()

		instance, err := backend.CreateInstance()
		if err != nil {
			return fmt.Errorf("CreateInstance failed: %w", err)
		}

		surface, err := backend.CreateSurface(instance, types.SurfaceHandle{Window: layer.Ptr()})
		if err != nil {
			return fmt.Errorf("CreateSurface failed: %w", err)
		}

		adapter, err := backend.RequestAdapter(instance, &types.AdapterOptions{
			PowerPreference: types.PowerPreferenceHighPerformance,
		})
		if err != nil {
			return fmt.Errorf("RequestAdapter failed: %w", err)
		}

		device, err := backend.RequestDevice(adapter, nil)
		if err != nil {
			return fmt.Errorf("RequestDevice failed: %w", err)
		}

		queue := backend.GetQueue(device)
		if queue == 0 {
			return fmt.Errorf("GetQueue returned 0")
		}

		backend.ConfigureSurface(surface, device, &types.SurfaceConfig{
			Format:      types.TextureFormatBGRA8Unorm,
			Usage:       types.TextureUsageRenderAttachment,
			Width:       64,
			Height:      64,
			AlphaMode:   types.AlphaModeOpaque,
			PresentMode: types.PresentModeFifo,
		})

		surfTex, err := acquireSurfaceTexture(backend, surface)
		if err != nil {
			return err
		}
		view := backend.CreateTextureView(surfTex.Texture, nil)
		if view == 0 {
			return fmt.Errorf("CreateTextureView returned 0")
		}
		defer func() {
			backend.ReleaseTextureView(view)
			backend.ReleaseTexture(surfTex.Texture)
		}()

		if err := writeTextureCases(backend, device, queue); err != nil {
			return err
		}

		sampleSize := types.Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1}
		texture, err := backend.CreateTexture(device, &types.TextureDescriptor{
			Label:         "sample-texture",
			Size:          sampleSize,
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     types.TextureDimension2D,
			Format:        types.TextureFormatRGBA8Unorm,
			Usage:         types.TextureUsageTextureBinding | types.TextureUsageCopyDst,
		})
		if err != nil {
			return fmt.Errorf("CreateTexture failed: %w", err)
		}
		textureView := backend.CreateTextureView(texture, nil)
		if textureView == 0 {
			return fmt.Errorf("CreateTextureView(texture) returned 0")
		}
		sampler, err := backend.CreateSampler(device, &types.SamplerDescriptor{
			Label:        "sample-sampler",
			AddressModeU: types.AddressModeClampToEdge,
			AddressModeV: types.AddressModeClampToEdge,
			AddressModeW: types.AddressModeClampToEdge,
			MagFilter:    types.FilterModeNearest,
			MinFilter:    types.FilterModeNearest,
			MipmapFilter: types.MipmapFilterModeNearest,
		})
		if err != nil {
			return fmt.Errorf("CreateSampler failed: %w", err)
		}
		defer func() {
			backend.ReleaseSampler(sampler)
			backend.ReleaseTextureView(textureView)
			backend.ReleaseTexture(texture)
		}()

		sampleBPP, ok := bytesPerPixel(types.TextureFormatRGBA8Unorm)
		if !ok {
			return fmt.Errorf("bytesPerPixel missing for RGBA8Unorm")
		}
		sampleRowBytes := alignTo256(sampleSize.Width * sampleBPP)
		textureData := make([]byte, int(sampleRowBytes*sampleSize.Height))
		for i := range textureData {
			textureData[i] = byte((i*31 + 17) & 0xFF)
		}
		backend.WriteTexture(queue, &types.ImageCopyTexture{
			Texture:  texture,
			MipLevel: 0,
			Origin:   types.Origin3D{},
			Aspect:   types.TextureAspectAll,
		}, textureData, &types.ImageDataLayout{
			Offset:       0,
			BytesPerRow:  sampleRowBytes,
			RowsPerImage: sampleSize.Height,
		}, &sampleSize)

		uniformData := f32Bytes(1.0, 1.0, 1.0, 1.0)
		uniformBuffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
			Label: "uniform-buffer",
			Size:  uint64(len(uniformData)),
			Usage: types.BufferUsageUniform | types.BufferUsageCopyDst,
		})
		if err != nil {
			return fmt.Errorf("CreateBuffer(uniform) failed: %w", err)
		}
		backend.WriteBuffer(queue, uniformBuffer, 0, uniformData)

		vertexData := f32Bytes(
			0.0, 0.5, 1.0, 0.0, 0.0,
			-0.5, -0.5, 0.0, 1.0, 0.0,
			0.5, -0.5, 0.0, 0.0, 1.0,
		)
		vertexBuffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
			Label: "vertex-buffer",
			Size:  uint64(len(vertexData)),
			Usage: types.BufferUsageVertex | types.BufferUsageCopyDst,
		})
		if err != nil {
			return fmt.Errorf("CreateBuffer(vertex) failed: %w", err)
		}
		backend.WriteBuffer(queue, vertexBuffer, 0, vertexData)

		indexData := u16Bytes(0, 1, 2)
		indexBuffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
			Label: "index-buffer",
			Size:  uint64(len(indexData)),
			Usage: types.BufferUsageIndex | types.BufferUsageCopyDst,
		})
		if err != nil {
			return fmt.Errorf("CreateBuffer(index) failed: %w", err)
		}
		backend.WriteBuffer(queue, indexBuffer, 0, indexData)
		defer func() {
			backend.ReleaseBuffer(indexBuffer)
			backend.ReleaseBuffer(vertexBuffer)
			backend.ReleaseBuffer(uniformBuffer)
		}()

		bindGroupLayout, err := backend.CreateBindGroupLayout(device, &types.BindGroupLayoutDescriptor{
			Label: "bind-group-layout",
			Entries: []types.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: types.ShaderStageFragment,
					Buffer: &types.BufferBindingLayout{
						Type:             types.BufferBindingTypeUniform,
						HasDynamicOffset: false,
						MinBindingSize:   16,
					},
				},
				{
					Binding:    1,
					Visibility: types.ShaderStageFragment,
					Sampler: &types.SamplerBindingLayout{
						Type: types.SamplerBindingTypeFiltering,
					},
				},
				{
					Binding:    2,
					Visibility: types.ShaderStageFragment,
					Texture: &types.TextureBindingLayout{
						SampleType:    types.TextureSampleTypeFloat,
						ViewDimension: types.TextureViewDimension2D,
						Multisampled:  false,
					},
				},
			},
		})
		if err != nil {
			return fmt.Errorf("CreateBindGroupLayout failed: %w", err)
		}
		defer backend.ReleaseBindGroupLayout(bindGroupLayout)

		bindGroup, err := backend.CreateBindGroup(device, &types.BindGroupDescriptor{
			Label:  "bind-group",
			Layout: bindGroupLayout,
			Entries: []types.BindGroupEntry{
				{
					Binding: 0,
					Buffer:  uniformBuffer,
					Offset:  0,
					Size:    uint64(len(uniformData)),
				},
				{
					Binding: 1,
					Sampler: sampler,
				},
				{
					Binding:     2,
					TextureView: textureView,
				},
			},
		})
		if err != nil {
			return fmt.Errorf("CreateBindGroup failed: %w", err)
		}
		defer backend.ReleaseBindGroup(bindGroup)

		pipelineLayout, err := backend.CreatePipelineLayout(device, &types.PipelineLayoutDescriptor{
			Label:            "pipeline-layout",
			BindGroupLayouts: []types.BindGroupLayout{bindGroupLayout},
		})
		if err != nil {
			return fmt.Errorf("CreatePipelineLayout failed: %w", err)
		}
		defer backend.ReleasePipelineLayout(pipelineLayout)

		shader, err := backend.CreateShaderModuleWGSL(device, backendTestWGSL)
		if err != nil {
			return fmt.Errorf("CreateShaderModuleWGSL failed: %w", err)
		}

		pipeline, err := backend.CreateRenderPipeline(device, &types.RenderPipelineDescriptor{
			Layout:           pipelineLayout,
			VertexShader:     shader,
			VertexEntryPoint: "vs_main",
			FragmentShader:   shader,
			FragmentEntry:    "fs_main",
			TargetFormat:     types.TextureFormatBGRA8Unorm,
			VertexBuffers: []types.VertexBufferLayout{
				{
					ArrayStride: 20,
					StepMode:    types.VertexStepModeVertex,
					Attributes: []types.VertexAttribute{
						{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
						{Format: types.VertexFormatFloat32x3, Offset: 8, ShaderLocation: 1},
					},
				},
			},
		})
		if err != nil {
			return fmt.Errorf("CreateRenderPipeline failed: %w", err)
		}

		encoder := backend.CreateCommandEncoder(device)
		if encoder == 0 {
			return fmt.Errorf("CreateCommandEncoder returned 0")
		}

		pass := backend.BeginRenderPass(encoder, &types.RenderPassDescriptor{
			ColorAttachments: []types.ColorAttachment{
				{
					View:       view,
					LoadOp:     types.LoadOpClear,
					StoreOp:    types.StoreOpStore,
					ClearValue: types.Color{R: 0.1, G: 0.2, B: 0.3, A: 1.0},
				},
			},
		})
		if pass == 0 {
			return fmt.Errorf("BeginRenderPass returned 0")
		}

		backend.SetPipeline(pass, pipeline)
		// TODO: Validate Metal bind group resource wiring once HAL argument buffers are implemented.
		backend.SetBindGroup(pass, 0, bindGroup, nil)
		backend.SetVertexBuffer(pass, 0, vertexBuffer, 0, uint64(len(vertexData)))
		backend.SetIndexBuffer(pass, indexBuffer, types.IndexFormatUint16, 0, uint64(len(indexData)))
		backend.DrawIndexed(pass, 3, 1, 0, 0, 0)

		backend.EndRenderPass(pass)
		backend.ReleaseRenderPass(pass)

		cmd := backend.FinishEncoder(encoder)
		backend.ReleaseCommandEncoder(encoder)
		if cmd == 0 {
			return fmt.Errorf("FinishEncoder returned 0")
		}

		backend.Submit(queue, cmd)
		backend.ReleaseCommandBuffer(cmd)
		backend.Present(surface)

		backend.ReleaseTexture(0)
		backend.ReleaseTextureView(0)
		backend.ReleaseSampler(0)
		backend.ReleaseBuffer(0)
		backend.ReleaseBindGroupLayout(0)
		backend.ReleaseBindGroup(0)
		backend.ReleasePipelineLayout(0)
		backend.ReleaseCommandBuffer(0)
		backend.ReleaseCommandEncoder(0)
		backend.ReleaseRenderPass(0)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func acquireSurfaceTexture(backend gpu.Backend, surface types.Surface) (types.SurfaceTexture, error) {
	var last types.SurfaceTexture
	var lastErr error
	for i := 0; i < 5; i++ {
		last, lastErr = backend.GetCurrentTexture(surface)
		if lastErr == nil && last.Status == types.SurfaceStatusSuccess && last.Texture != 0 {
			return last, nil
		}
		time.Sleep(10 * time.Millisecond)
	}

	if lastErr != nil {
		return types.SurfaceTexture{}, fmt.Errorf("GetCurrentTexture failed: %w (status=%v)", lastErr, last.Status)
	}
	if last.Status != types.SurfaceStatusSuccess {
		return types.SurfaceTexture{}, fmt.Errorf("GetCurrentTexture status=%v", last.Status)
	}
	return types.SurfaceTexture{}, fmt.Errorf("GetCurrentTexture returned texture=0")
}

func f32Bytes(values ...float32) []byte {
	out := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
	}
	return out
}

func newMetalLayer() (*darwin.MetalLayer, error) {
	layer, err := darwin.NewMetalLayer()
	if err != nil {
		return nil, fmt.Errorf("NewMetalLayer failed: %w", err)
	}
	return layer, nil
}

func u16Bytes(values ...uint16) []byte {
	out := make([]byte, 2*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint16(out[i*2:], value)
	}
	return out
}

func alignTo256(value uint32) uint32 {
	if value == 0 {
		return 0
	}
	const alignment = 256
	return (value + alignment - 1) &^ (alignment - 1)
}

func bytesPerPixel(format types.TextureFormat) (uint32, bool) {
	switch format {
	case types.TextureFormatRGBA8Unorm, types.TextureFormatBGRA8Unorm:
		return 4, true
	default:
		return 0, false
	}
}

func writeTextureCases(backend gpu.Backend, device types.Device, queue types.Queue) error {
	type textureCase struct {
		label       string
		format      types.TextureFormat
		size        types.Extent3D
		bytesPerRow uint32
	}

	cases := []textureCase{
		{
			label:  "rgba8-128x64",
			format: types.TextureFormatRGBA8Unorm,
			size:   types.Extent3D{Width: 128, Height: 64, DepthOrArrayLayers: 1},
		},
		{
			label:       "rgba8-65x32-padded",
			format:      types.TextureFormatRGBA8Unorm,
			size:        types.Extent3D{Width: 65, Height: 32, DepthOrArrayLayers: 1},
			bytesPerRow: 1024,
		},
		{
			label:  "bgra8-320x8",
			format: types.TextureFormatBGRA8Unorm,
			size:   types.Extent3D{Width: 320, Height: 8, DepthOrArrayLayers: 1},
		},
		{
			label:  "bgra8-96x48",
			format: types.TextureFormatBGRA8Unorm,
			size:   types.Extent3D{Width: 96, Height: 48, DepthOrArrayLayers: 1},
		},
	}

	for _, tc := range cases {
		bpp, ok := bytesPerPixel(tc.format)
		if !ok {
			return fmt.Errorf("bytesPerPixel missing for %v", tc.format)
		}
		rowBytes := tc.size.Width * bpp
		bytesPerRow := tc.bytesPerRow
		if bytesPerRow == 0 {
			bytesPerRow = alignTo256(rowBytes)
		}
		if bytesPerRow < rowBytes || (bytesPerRow%256) != 0 {
			return fmt.Errorf("invalid bytesPerRow %d for %s (rowBytes=%d)", bytesPerRow, tc.label, rowBytes)
		}

		texture, err := backend.CreateTexture(device, &types.TextureDescriptor{
			Label:         "write-" + tc.label,
			Size:          tc.size,
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     types.TextureDimension2D,
			Format:        tc.format,
			Usage:         types.TextureUsageCopyDst,
		})
		if err != nil {
			return fmt.Errorf("CreateTexture(%s) failed: %w", tc.label, err)
		}

		data := make([]byte, int(bytesPerRow*tc.size.Height))
		for i := range data {
			data[i] = byte((i*13 + 7) & 0xFF)
		}

		backend.WriteTexture(queue, &types.ImageCopyTexture{
			Texture:  texture,
			MipLevel: 0,
			Origin:   types.Origin3D{},
			Aspect:   types.TextureAspectAll,
		}, data, &types.ImageDataLayout{
			Offset:       0,
			BytesPerRow:  bytesPerRow,
			RowsPerImage: tc.size.Height,
		}, &tc.size)

		backend.ReleaseTexture(texture)
	}
	return nil
}

var mainThread = make(chan func())

func TestMain(m *testing.M) {
	runtime.LockOSThread()

	done := make(chan int, 1)
	go func() { done <- m.Run() }()

	for {
		select {
		case fn := <-mainThread:
			fn()
		case code := <-done:
			os.Exit(code)
		}
	}
}

func runOnMainThread(fn func() error) error {
	done := make(chan error, 1)
	mainThread <- func() { done <- fn() }
	return <-done
}
