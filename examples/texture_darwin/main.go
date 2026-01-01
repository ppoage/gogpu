//go:build darwin

package main

import (
	_ "embed"
	"encoding/binary"
	"log"
	"math"

	"github.com/gogpu/gogpu/gpu/backend/native"
	"github.com/gogpu/gogpu/gpu/types"
	"github.com/gogpu/gogpu/internal/platform"
)

//go:embed shader.wgsl
var texturedWGSL string

func main() {
	plat := platform.New()
	if err := plat.Init(platform.Config{
		Title:     "GoGPU - Textured Quad (Darwin)",
		Width:     800,
		Height:    600,
		Resizable: true,
	}); err != nil {
		log.Fatal(err)
	}
	defer plat.Destroy()

	backend := native.New()
	if err := backend.Init(); err != nil {
		log.Fatal(err)
	}
	defer backend.Destroy()

	instance, err := backend.CreateInstance()
	if err != nil {
		log.Fatal(err)
	}

	hinstance, hwnd := plat.GetHandle()
	surface, err := backend.CreateSurface(instance, types.SurfaceHandle{
		Instance: hinstance,
		Window:   hwnd,
	})
	if err != nil {
		log.Fatal(err)
	}

	adapter, err := backend.RequestAdapter(instance, &types.AdapterOptions{
		PowerPreference: types.PowerPreferenceHighPerformance,
	})
	if err != nil {
		log.Fatal(err)
	}

	device, err := backend.RequestDevice(adapter, nil)
	if err != nil {
		log.Fatal(err)
	}

	queue := backend.GetQueue(device)
	format := types.TextureFormatBGRA8Unorm
	surfaceConfigured := configureSurface(backend, plat, surface, device, format)

	texture, textureView, sampler := createTextureResources(backend, device, queue)
	defer backend.ReleaseSampler(sampler)
	defer backend.ReleaseTextureView(textureView)
	defer backend.ReleaseTexture(texture)

	bindGroupLayout, err := backend.CreateBindGroupLayout(device, &types.BindGroupLayoutDescriptor{
		Label: "texture-bind-group-layout",
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageFragment,
				Sampler: &types.SamplerBindingLayout{
					Type: types.SamplerBindingTypeFiltering,
				},
			},
			{
				Binding:    1,
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
		log.Fatal(err)
	}
	defer backend.ReleaseBindGroupLayout(bindGroupLayout)

	pipelineLayout, err := backend.CreatePipelineLayout(device, &types.PipelineLayoutDescriptor{
		Label:            "texture-pipeline-layout",
		BindGroupLayouts: []types.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer backend.ReleasePipelineLayout(pipelineLayout)

	bindGroup, err := backend.CreateBindGroup(device, &types.BindGroupDescriptor{
		Label:  "texture-bind-group",
		Layout: bindGroupLayout,
		Entries: []types.BindGroupEntry{
			{Binding: 0, Sampler: sampler},
			{Binding: 1, TextureView: textureView},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer backend.ReleaseBindGroup(bindGroup)

	shader, err := backend.CreateShaderModuleWGSL(device, texturedWGSL)
	if err != nil {
		log.Fatal(err)
	}

	pipeline, err := backend.CreateRenderPipeline(device, &types.RenderPipelineDescriptor{
		Label:            "textured-quad-pipeline",
		Layout:           pipelineLayout,
		VertexShader:     shader,
		VertexEntryPoint: "vs_main",
		FragmentShader:   shader,
		FragmentEntry:    "fs_main",
		TargetFormat:     format,
		Topology:         types.PrimitiveTopologyTriangleList,
		FrontFace:        types.FrontFaceCCW,
		CullMode:         types.CullModeNone,
		VertexBuffers: []types.VertexBufferLayout{
			{
				ArrayStride: 16,
				StepMode:    types.VertexStepModeVertex,
				Attributes: []types.VertexAttribute{
					{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
					{Format: types.VertexFormatFloat32x2, Offset: 8, ShaderLocation: 1},
				},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	vertexBuffer, indexBuffer := createGeometryBuffers(backend, device, queue)
	defer backend.ReleaseBuffer(vertexBuffer)
	defer backend.ReleaseBuffer(indexBuffer)

	for !plat.ShouldClose() {
		for {
			event := plat.PollEvents()
			if event.Type == platform.EventNone {
				break
			}
			if event.Type == platform.EventResize {
				surfaceConfigured = configureSurface(backend, plat, surface, device, format)
			}
		}

		if !surfaceConfigured {
			continue
		}

		surfaceTex, err := backend.GetCurrentTexture(surface)
		if err != nil || surfaceTex.Status != types.SurfaceStatusSuccess || surfaceTex.Texture == 0 {
			if configureSurface(backend, plat, surface, device, format) {
				surfaceConfigured = true
			}
			continue
		}

		view := backend.CreateTextureView(surfaceTex.Texture, nil)
		if view == 0 {
			backend.ReleaseTexture(surfaceTex.Texture)
			continue
		}

		encoder := backend.CreateCommandEncoder(device)
		if encoder == 0 {
			backend.ReleaseTextureView(view)
			backend.ReleaseTexture(surfaceTex.Texture)
			continue
		}

		pass := backend.BeginRenderPass(encoder, &types.RenderPassDescriptor{
			ColorAttachments: []types.ColorAttachment{
				{
					View:       view,
					LoadOp:     types.LoadOpClear,
					StoreOp:    types.StoreOpStore,
					ClearValue: types.Color{R: 0.1, G: 0.1, B: 0.12, A: 1.0},
				},
			},
		})
		if pass == 0 {
			backend.ReleaseCommandEncoder(encoder)
			backend.ReleaseTextureView(view)
			backend.ReleaseTexture(surfaceTex.Texture)
			continue
		}

		backend.SetPipeline(pass, pipeline)
		backend.SetBindGroup(pass, 0, bindGroup, nil)
		backend.SetVertexBuffer(pass, 0, vertexBuffer, 0, 0)
		backend.SetIndexBuffer(pass, indexBuffer, types.IndexFormatUint16, 0, 0)
		backend.DrawIndexed(pass, 6, 1, 0, 0, 0)
		backend.EndRenderPass(pass)
		backend.ReleaseRenderPass(pass)

		cmd := backend.FinishEncoder(encoder)
		backend.ReleaseCommandEncoder(encoder)
		if cmd != 0 {
			backend.Submit(queue, cmd)
			backend.ReleaseCommandBuffer(cmd)
		}

		backend.Present(surface)
		backend.ReleaseTextureView(view)
		backend.ReleaseTexture(surfaceTex.Texture)
	}
}

func configureSurface(backend *native.Backend, plat platform.Platform, surface types.Surface, device types.Device, format types.TextureFormat) bool {
	width, height := plat.GetSize()
	if width <= 0 || height <= 0 {
		return false
	}

	backend.ConfigureSurface(surface, device, &types.SurfaceConfig{
		Format:      format,
		Usage:       types.TextureUsageRenderAttachment,
		Width:       uint32(width),
		Height:      uint32(height),
		AlphaMode:   types.AlphaModeOpaque,
		PresentMode: types.PresentModeFifo,
	})
	return true
}

func createGeometryBuffers(backend *native.Backend, device types.Device, queue types.Queue) (types.Buffer, types.Buffer) {
	vertices := []float32{
		-0.75, -0.75, 0.0, 1.0,
		0.75, -0.75, 1.0, 1.0,
		0.75, 0.75, 1.0, 0.0,
		-0.75, 0.75, 0.0, 0.0,
	}
	indices := []uint16{0, 1, 2, 0, 2, 3}

	vertexBytes := packFloat32s(vertices)
	indexBytes := packUint16s(indices)

	vertexBuffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
		Label: "quad-vertex-buffer",
		Size:  uint64(len(vertexBytes)),
		Usage: types.BufferUsageVertex | types.BufferUsageCopyDst,
	})
	if err != nil {
		log.Fatal(err)
	}
	backend.WriteBuffer(queue, vertexBuffer, 0, vertexBytes)

	indexBuffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
		Label: "quad-index-buffer",
		Size:  uint64(len(indexBytes)),
		Usage: types.BufferUsageIndex | types.BufferUsageCopyDst,
	})
	if err != nil {
		log.Fatal(err)
	}
	backend.WriteBuffer(queue, indexBuffer, 0, indexBytes)

	return vertexBuffer, indexBuffer
}

func createTextureResources(backend *native.Backend, device types.Device, queue types.Queue) (types.Texture, types.TextureView, types.Sampler) {
	const texWidth = 256
	const texHeight = 256

	pixels := makeCheckerboard(texWidth, texHeight)

	texture, err := backend.CreateTexture(device, &types.TextureDescriptor{
		Label: "checkerboard-texture",
		Size: types.Extent3D{
			Width:              texWidth,
			Height:             texHeight,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     types.TextureDimension2D,
		Format:        types.TextureFormatRGBA8Unorm,
		Usage:         types.TextureUsageCopyDst | types.TextureUsageTextureBinding,
	})
	if err != nil {
		log.Fatal(err)
	}

	backend.WriteTexture(queue,
		&types.ImageCopyTexture{Texture: texture},
		pixels,
		&types.ImageDataLayout{
			Offset:       0,
			BytesPerRow:  texWidth * 4,
			RowsPerImage: texHeight,
		},
		&types.Extent3D{
			Width:              texWidth,
			Height:             texHeight,
			DepthOrArrayLayers: 1,
		},
	)

	view := backend.CreateTextureView(texture, nil)
	if view == 0 {
		log.Fatal("failed to create texture view")
	}

	sampler, err := backend.CreateSampler(device, &types.SamplerDescriptor{
		Label:        "texture-sampler",
		AddressModeU: types.AddressModeClampToEdge,
		AddressModeV: types.AddressModeClampToEdge,
		AddressModeW: types.AddressModeClampToEdge,
		MagFilter:    types.FilterModeLinear,
		MinFilter:    types.FilterModeLinear,
		MipmapFilter: types.MipmapFilterModeNearest,
	})
	if err != nil {
		log.Fatal(err)
	}

	return texture, view, sampler
}

func makeCheckerboard(width, height uint32) []byte {
	pixels := make([]byte, int(width*height*4))
	for y := uint32(0); y < height; y++ {
		for x := uint32(0); x < width; x++ {
			i := int((y*width + x) * 4)
			if ((x/32)+(y/32))%2 == 0 {
				pixels[i] = 255
				pixels[i+1] = 255
				pixels[i+2] = 255
				pixels[i+3] = 255
			} else {
				pixels[i] = 30
				pixels[i+1] = 144
				pixels[i+2] = 255
				pixels[i+3] = 255
			}
		}
	}
	return pixels
}

func packFloat32s(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func packUint16s(values []uint16) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], v)
	}
	return out
}
