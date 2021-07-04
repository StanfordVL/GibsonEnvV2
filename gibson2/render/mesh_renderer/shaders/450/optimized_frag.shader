#version 450

#define MAX_ARRAY_SIZE 1024

uniform sampler2DArray bigTex;
uniform sampler2DArray smallTex;

uniform float use_two_light_probes;

uniform samplerCube specularTexture;
uniform samplerCube irradianceTexture;
uniform sampler2D specularBRDF_LUT;

uniform samplerCube specularTexture2;
uniform samplerCube irradianceTexture2;
uniform sampler2D specularBRDF_LUT2;

uniform sampler2D lightModulationMap;

uniform vec3 eyePosition;

uniform sampler2D defaultMetallicTexture;
uniform sampler2D defaultRoughnessTexture;
uniform sampler2D defaultNormalTexture;

layout (std140) uniform TexColorData {
    vec4 tex_data[MAX_ARRAY_SIZE];
    vec4 tex_roughness_metallic_data[MAX_ARRAY_SIZE];
    vec4 tex_normal_data[MAX_ARRAY_SIZE];
    vec4 diffuse_colors[MAX_ARRAY_SIZE];
};

layout (std140) uniform PBRData {
    vec4 pbr_data[MAX_ARRAY_SIZE];
};

uniform sampler2D depthMap;
uniform int shadow_pass;

in vec2 theCoords;
in vec3 Normal_world;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Pos_cam_prev;
in vec3 Pos_cam_projected;
in vec3 Diffuse_color;
in mat3 TBN;
in vec4 FragPosLightSpace;
flat in int Draw_id;
in vec2 Optical_flow;

const float PI = 3.141592;
const float Epsilon = 0.00001;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;
layout (location = 4) out vec4 SceneFlowColour;
layout (location = 5) out vec4 OpticalFlowColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color
float ndfGGX(float cosLh, float roughness)
{
	float alpha   = roughness * roughness;
	float alphaSq = alpha * alpha;

	float denom = (cosLh * cosLh) * (alphaSq - 1.0) + 1.0;
	return alphaSq / (PI * denom * denom);
}

// Single term for separable Schlick-GGX below.
float gaSchlickG1(float cosTheta, float k)
{
	return cosTheta / (cosTheta * (1.0 - k) + k);
}

// Schlick-GGX approximation of geometric attenuation function using Smith's method.
float gaSchlickGGX(float cosLi, float cosLo, float roughness)
{
	float r = roughness + 1.0;
	float k = (r * r) / 8.0; // Epic suggests using this roughness remapping for analytic lights.
	return gaSchlickG1(cosLi, k) * gaSchlickG1(cosLo, k);
}

// Shlick's approximation of the Fresnel factor.
vec3 fresnelSchlick(vec3 F0, float cosTheta)
{
	return F0 + (vec3(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * light_color;
    vec3 lightDir = vec3(0, 0, 1);
    float diff = 0.5 + 0.5 * max(dot(Normal_world, lightDir), 0.0);
    vec3 diffuse = diff * light_color;
    vec4 curr_tex_data = tex_data[Draw_id];
    int tex_num = int(curr_tex_data.x);
    int tex_layer = int(curr_tex_data.y);
    float instance_color = curr_tex_data.z;
    vec4 curr_pbr_data = pbr_data[Draw_id];
    int use_pbr = int(curr_pbr_data.x);
    vec2 texelSize = 1.0 / textureSize(depthMap, 0);

    float shadow = 0.0;

    if (shadow_pass == 2) {
        vec3 projCoords = FragPosLightSpace.xyz / FragPosLightSpace.w;
        projCoords = projCoords * 0.5 + 0.5;

        float cosTheta = dot(Normal_world, lightDir);
        cosTheta = clamp(cosTheta, 0.0, 1.0);
        float bias = 0.005*tan(acos(cosTheta));
        bias = clamp(bias, 0.001 ,0.1);
        float currentDepth = projCoords.z;
        float closestDepth = 0;

        shadow = 0.0;
        float current_shadow = 0;

        for(int x = -2; x <= 2; ++x)
        {
            for (int y = -2; y <= 2; ++y)
            {
                closestDepth = texture(depthMap, projCoords.xy + vec2(x, y) * texelSize).b * 0.5 + 0.5;
                current_shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
                if ((projCoords.z > 1.0) || (projCoords.x > 1.0) || (projCoords.y > 1.0)
                || (projCoords.x < 0) || (projCoords.y < 0)) current_shadow = 0.0;
                shadow += current_shadow;
            }
        }
        shadow /= 25.0;
    }
    else {
        shadow = 0.0;
    }

    if (use_pbr == 1) {
        int normal_tex_num = int(tex_normal_data[Draw_id].x);
        int normal_tex_layer = int(tex_normal_data[Draw_id].y);
        vec3 normal_map;
        if (normal_tex_num == -1) {
            normal_map = 2 * texture(defaultNormalTexture, theCoords).rgb - 1;
        } else if (normal_tex_num == 0) {
            normal_map = 2 * texture(bigTex, vec3(fract(theCoords.x), fract(theCoords.y), normal_tex_layer)).rgb - 1;
        } else {
            normal_map = 2 * texture(smallTex, vec3(fract(theCoords.x), fract(theCoords.y), normal_tex_layer)).rgb - 1;
        }
        vec3 N = normalize(TBN * normal_map);
        vec3 Lo = normalize(eyePosition - FragPos);
        float cosLo = max(0.0, dot(N, Lo));
        vec3 Lr = 2.0 * cosLo * N - Lo;

        vec3 albedo;
        if (tex_num == -1) {
            albedo = diffuse_colors[Draw_id].rgb;//diffuse color
        } else if (tex_num == 0) {
            albedo = texture(bigTex, vec3(fract(theCoords.x), fract(theCoords.y), tex_layer)).rgb;
        } else if (tex_num == 1) {
            albedo = texture(smallTex, vec3(fract(theCoords.x), fract(theCoords.y), tex_layer)).rgb;
        }

        int roughness_tex_num = int(tex_roughness_metallic_data[Draw_id].x);
        int roughness_tex_layer = int(tex_roughness_metallic_data[Draw_id].y);
        float roughness_sampled;
        if (roughness_tex_num == -1) {
            roughness_sampled = texture(defaultRoughnessTexture, theCoords).r;
        } else if (roughness_tex_num == 0) {
            roughness_sampled =  texture(bigTex, vec3(fract(theCoords.x), fract(theCoords.y), roughness_tex_layer)).r;
        } else {
            roughness_sampled = texture(smallTex, vec3(fract(theCoords.x), fract(theCoords.y), roughness_tex_layer)).r;
        }

        int metallic_tex_num = int(tex_roughness_metallic_data[Draw_id].z);
        int metallic_tex_layer = int(tex_roughness_metallic_data[Draw_id].w);
        float metallic_sampled;
        if (metallic_tex_num == -1) {
            metallic_sampled = texture(defaultMetallicTexture, theCoords).r;
        } else if (metallic_tex_num == 0) {
            metallic_sampled = texture(bigTex, vec3(fract(theCoords.x), fract(theCoords.y), metallic_tex_layer)).r;
        } else {
            metallic_sampled = texture(smallTex, vec3(fract(theCoords.x), fract(theCoords.y), metallic_tex_layer)).r;
        }

        vec3 Fdielectric = vec3(0.04);
        vec3 F0 = mix(Fdielectric, albedo, metallic_sampled);
        vec3 irradiance = texture(irradianceTexture, N).rgb;
        vec3 F = fresnelSchlick(F0, cosLo);
        vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metallic_sampled);
        vec3 diffuseIBL = kd * albedo * irradiance;
        int specularTextureLevels = textureQueryLevels(specularTexture);
        vec3 specularIrradiance = textureLod(specularTexture, Lr, roughness_sampled * specularTextureLevels).rgb;
        vec2 specularBRDF = texture(specularBRDF_LUT, vec2(cosLo, roughness_sampled)).rg;
        vec3 specularIBL = (F0 * specularBRDF.x + specularBRDF.y) * specularIrradiance;
        vec3 ambientLighting = diffuseIBL + specularIBL;
        //vec3 reflection = textureLod(specularTexture, vec3(Lr.x, Lr.z, Lr.y), 1).rgb;
        outputColour = vec4(ambientLighting, 1);

    }
    else {
        if (tex_num == -1) {
            outputColour = diffuse_colors[Draw_id] * diff; //diffuse color
        } else if (tex_num == 0) {
            outputColour = texture(bigTex, vec3(fract(theCoords.x), fract(theCoords.y), tex_layer));
        } else if (tex_num == 1) {
            outputColour = texture(smallTex, vec3(fract(theCoords.x), fract(theCoords.y), tex_layer));
        }
    }
    NormalColour =  vec4((Normal_cam + 1) / 2,1);
    InstanceColour = vec4(Instance_color,1);
    if (shadow_pass == 1) {
        PCColour = vec4(Pos_cam_projected, 1);
    } else {
        PCColour = vec4(Pos_cam, 1);
    }
    outputColour = outputColour *  (1 - shadow * 0.5);
    SceneFlowColour =  vec4(Pos_cam - Pos_cam_prev,1);
    OpticalFlowColour =  vec4(Optical_flow,0,1);
}