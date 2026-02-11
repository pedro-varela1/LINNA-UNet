"""
Visualização de Cobertura Esférica da Imagem no Dataset Lunar

Este script pega uma imagem de exemplo do dataset, extrai suas coordenadas
geográficas (latitude e longitude das bordas) e altitude, e cria uma 
visualização 3D de uma esfera representando a Lua com a região coberta
pela imagem pintada em verde.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import OpenEXR
import Imath

# Constantes
MOON_RADIUS_KM = 1737.4

def read_exr_channels(exr_path):
    """
    Lê os canais R e G de um arquivo EXR (latitude e longitude).
    """
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr_file.channel('R', FLOAT)
    g_str = exr_file.channel('G', FLOAT)
    
    lat_map = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
    lon_map = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
    
    return lat_map, lon_map

def latlon_to_cartesian(lat, lon, alt):
    """
    Converte coordenadas geográficas para cartesianas 3D.
    """
    r = MOON_RADIUS_KM + alt
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def create_sphere(radius, resolution=50):
    """
    Cria uma malha esférica para visualização.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    return x, y, z

def get_image_bounds(lat_map, lon_map):
    """
    Extrai os limites geográficos da imagem (bordas).
    """
    bounds = {
        'lat_min': lat_map.min(),
        'lat_max': lat_map.max(),
        'lon_min': lon_map.min(),
        'lon_max': lon_map.max(),
        'lat_center': lat_map.mean(),
        'lon_center': lon_map.mean()
    }
    return bounds

def visualize_moon_coverage(dataset_root, sample_idx=0):
    """
    Cria uma visualização 3D mostrando a região da Lua coberta por uma imagem.
    """
    # 1. Carregar arquivos
    img_files = sorted(glob.glob(os.path.join(dataset_root, "img", "*.png")))
    exr_files = sorted(glob.glob(os.path.join(dataset_root, "lat_lon_exr", "*.exr")))
    alt_files = sorted(glob.glob(os.path.join(dataset_root, "altimeter", "*.txt")))
    
    if not img_files or not exr_files or not alt_files:
        print("❌ Erro: Dataset não encontrado ou incompleto!")
        return
    
    # Usar apenas arquivos correspondentes
    img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
    exr_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in exr_files}
    alt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in alt_files}
    
    common_names = sorted(set(img_dict.keys()) & set(exr_dict.keys()) & set(alt_dict.keys()))
    
    if sample_idx >= len(common_names):
        print(f"❌ Erro: Índice {sample_idx} fora do range. Existem {len(common_names)} amostras.")
        return
    
    sample_name = common_names[sample_idx]
    img_path = img_dict[sample_name]
    exr_path = exr_dict[sample_name]
    alt_path = alt_dict[sample_name]
    
    # 2. Carregar dados
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    lat_map, lon_map = read_exr_channels(exr_path)
    
    with open(alt_path, 'r') as f:
        altitude = float(f.read().strip())
    
    # 3. Extrair informações das bordas
    bounds = get_image_bounds(lat_map, lon_map)
    
    print("=" * 60)
    print(f"📷 Amostra: {sample_name}")
    print(f"   Resolução da Imagem: {image.shape}")
    print("=" * 60)
    print("\n📍 COORDENADAS GEOGRÁFICAS:")
    print(f"   Latitude:")
    print(f"      Min: {bounds['lat_min']:.4f}°")
    print(f"      Max: {bounds['lat_max']:.4f}°")
    print(f"      Centro: {bounds['lat_center']:.4f}°")
    print(f"   Longitude:")
    print(f"      Min: {bounds['lon_min']:.4f}°")
    print(f"      Max: {bounds['lon_max']:.4f}°")
    print(f"      Centro: {bounds['lon_center']:.4f}°")
    print(f"\n🏔️  Altitude: {altitude:.2f} km")
    print("=" * 60)
    
    # 4. Criar visualização 3D
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Imagem Original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Imagem Original\n{sample_name}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Subplot 2: Mapa de Latitude
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(lat_map, cmap='viridis')
    ax2.set_title('Mapa de Latitude (°)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    # Subplot 3: Visualização 3D da Esfera
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Criar esfera base (Lua) em cinza claro
    sphere_radius = MOON_RADIUS_KM + altitude
    x_sphere, y_sphere, z_sphere = create_sphere(MOON_RADIUS_KM, resolution=50)
    ax3.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, 
                     edgecolor='none', linewidth=0)
    
    # Criar malha da região coberta pela imagem
    # Subsampling para não sobrecarregar a visualização
    step = max(1, lat_map.shape[0] // 50)  # Aproximadamente 50 pontos por dimensão
    lat_sub = lat_map[::step, ::step]
    lon_sub = lon_map[::step, ::step]
    
    # Converter pontos da imagem para coordenadas cartesianas
    x_img, y_img, z_img = latlon_to_cartesian(lat_sub, lon_sub, altitude)
    
    # Plotar a região coberta em verde
    ax3.plot_surface(x_img, y_img, z_img, color='green', alpha=0.8, 
                     edgecolor='darkgreen', linewidth=0.5)
    
    # Configurar visualização
    ax3.set_xlabel('X (km)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Y (km)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Z (km)', fontsize=10, fontweight='bold')
    ax3.set_title('Cobertura na Esfera Lunar\n(Verde = Região da Imagem)', 
                  fontsize=12, fontweight='bold')
    
    # Ajustar limites dos eixos para mostrar a esfera completa
    max_range = sphere_radius * 1.1
    ax3.set_xlim([-max_range, max_range])
    ax3.set_ylim([-max_range, max_range])
    ax3.set_zlim([-max_range, max_range])
    
    # Aspecto igual para manter a esfera circular
    ax3.set_box_aspect([1, 1, 1])
    
    # Ajustar ângulo de visualização para melhor perspectiva
    ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Salvar figura
    output_path = f"visualization_{sample_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualização salva em: {output_path}")
    
    plt.show()

def main():
    """
    Função principal - executa a visualização.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualizar cobertura esférica de imagem lunar')
    parser.add_argument('--dataset', type=str, default='./dataset/LunarLanding_Dataset',
                        help='Caminho para o dataset')
    parser.add_argument('--sample', type=int, default=10000,
                        help='Índice da amostra a visualizar (padrão: 10000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Erro: Dataset não encontrado em {args.dataset}")
        print("   Ajuste o caminho usando --dataset <caminho>")
        return
    
    visualize_moon_coverage(args.dataset, args.sample)

if __name__ == "__main__":
    main()
