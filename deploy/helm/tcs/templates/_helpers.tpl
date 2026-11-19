{{- define "tcs.name" -}}
tcs-api-{{ .Values.pool }}
{{- end -}}

{{- define "tcs.labels" -}}
app.kubernetes.io/name: tcs-api
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
tcs/pool: {{ .Values.pool }}
{{- end -}}
