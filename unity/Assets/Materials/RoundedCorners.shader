Shader "UI/RoundedCorners"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _Radius ("Corner Radius", Range(0, 0.5)) = 0.1
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent" }
        Cull Off
        Lighting Off
        ZWrite Off
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            fixed4 _Color;
            float _Radius;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Assume the rect's UVs go from 0 to 1.
                float2 uv = i.uv;
                
                // Calculate how far the pixel is from the inner rectangle.
                float2 dist = abs(uv - 0.5) - (0.5 - _Radius);
                float delta = length(max(dist, 0.0));
                
                // Discard pixels outside the corner radius.
                if(delta > _Radius)
                    discard;

                return _Color;
            }
            ENDCG
        }
    }
}
