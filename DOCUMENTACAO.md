# Face Security — Documentação do MVP

## 1. Visão geral do projeto

O **Face Security** é um projeto acadêmico com foco em reconhecimento facial para apoio à segurança residencial. O objetivo do MVP é demonstrar, de forma simples e funcional, um sistema capaz de abrir a câmera do computador, identificar se uma pessoa está cadastrada na base local e classificar a ocorrência como situação normal ou de atenção.

O sistema não tem o objetivo de rotular pessoas, prever incidentes ou tomar decisões automáticas contra pessoas. A proposta correta para apresentação é:

> O Face Security identifica pessoas cadastradas e não cadastradas em uma área monitorada, auxiliando na prevenção de incidentes por meio de alertas visuais e registros de evidência.

---

## 2. Objetivo do MVP

Criar uma versão simples e funcional do sistema para ser gravada em vídeo e apresentada ao professor.

O MVP deve conseguir:

1. Abrir a webcam do computador.
2. Capturar imagens em tempo real.
3. Comparar o rosto da pessoa com fotos cadastradas.
4. Exibir na tela se a pessoa foi reconhecida ou não.
5. Classificar o nível de atenção.
6. Salvar evidências de pessoas não reconhecidas.
7. Permitir gravação de vídeo demonstrando o funcionamento.

---

## 3. Escopo do MVP

### O que o MVP terá

- Captura de vídeo pela webcam.
- Cadastro simples de pessoas por imagem em uma pasta local.
- Reconhecimento facial usando biblioteca pronta.
- Classificação simples de risco.
- Alerta visual na tela.
- Salvamento de imagem quando uma pessoa não cadastrada for detectada.
- Estrutura de código em monolito modular.

### O que o MVP não terá nesta primeira versão

- Login de usuário.
- Aplicativo mobile.
- Envio real de alerta por WhatsApp, SMS ou e-mail.
- Banco de dados complexo.
- Consulta de antecedentes externos.
- Integração com câmeras externas.
- Deploy em nuvem.
- Sistema definitivo de vigilância.

Essas funcionalidades podem ser apresentadas como melhorias futuras.

---

## 4. Tecnologias recomendadas

Para o MVP, a recomendação é usar tecnologias simples e rápidas de implementar.

| Tecnologia | Função no projeto |
|---|---|
| Python | Linguagem principal do sistema |
| OpenCV | Captura da webcam e exibição do vídeo |
| DeepFace | Reconhecimento facial com modelos prontos |
| NumPy | Apoio ao processamento de imagens |
| SQLite | Registro simples de eventos, se necessário |
| GitHub | Controle de versão do código |
| Google Drive | Armazenamento de evidências, vídeos e documentos |
| Markdown | Documentação do projeto |

### Por que usar DeepFace no MVP?

A biblioteca DeepFace facilita o reconhecimento facial porque já traz modelos prontos. Assim, a equipe não precisa treinar uma inteligência artificial do zero.

Com ela, o grupo pode focar em:

- abrir a câmera;
- comparar rostos;
- mostrar resultados;
- salvar evidências;
- documentar o funcionamento.

### E o Hugging Face?

O Hugging Face é uma plataforma muito boa para encontrar modelos de inteligência artificial, inclusive modelos relacionados a visão computacional. Porém, para este MVP, ele pode deixar o projeto mais complexo.

Para a primeira entrega, a recomendação é:

> Usar OpenCV + DeepFace para o MVP.

O Hugging Face pode ser citado como evolução futura:

> Em versões futuras, o Face Security poderá utilizar modelos obtidos por meio do Hugging Face para testar arquiteturas mais avançadas de reconhecimento facial e melhorar a precisão do sistema.

---

## 5. Arquitetura adotada

O projeto será desenvolvido como um **monolito modular**.

Isso significa que o sistema será uma única aplicação Python, mas com o código separado em módulos por responsabilidade.

### Justificativa da arquitetura

O monolito modular foi escolhido porque:

- é mais simples para um projeto acadêmico;
- facilita a divisão de tarefas entre os integrantes;
- permite organizar melhor o código;
- evita a complexidade de microsserviços;
- facilita a gravação do funcionamento;
- permite evolução futura para módulos mais independentes.

Descrição para usar nos slides:

> O Face Security utiliza uma arquitetura de monolito modular, na qual a aplicação é mantida em um único projeto Python, porém organizada em módulos separados por responsabilidade, como captura de câmera, reconhecimento facial, análise de risco, alertas, logs e documentação.

---

## 6. Estrutura de pastas do projeto

```bash
face-security/
│
├── README.md
├── requirements.txt
├── main.py
│
├── app/
│   ├── __init__.py
│   ├── camera.py
│   ├── face_recognition_service.py
│   ├── risk_analyzer.py
│   ├── alert_service.py
│   └── logger.py
│
├── data/
│   ├── known_faces/
│   │   ├── integrante_1.jpg
│   │   └── integrante_2.jpg
│   │
│   ├── unknown_faces/
│   └── logs/
```

---

## 7. Explicação das pastas

### `main.py`

Arquivo principal do sistema. Ele inicia a aplicação, abre a câmera e chama os módulos responsáveis pelo reconhecimento, análise de risco e alerta.

### `app/`

Pasta onde ficará o código principal do sistema.

### `app/camera.py`

Responsável por abrir a webcam e capturar os frames de vídeo.

### `app/face_recognition_service.py`

Responsável por comparar o rosto capturado com as imagens cadastradas na pasta `data/known_faces`.

### `app/risk_analyzer.py`

Responsável por definir o nível de atenção da ocorrência.

Exemplo:

- Pessoa reconhecida: baixo risco.
- Pessoa não reconhecida: atenção.
- Pessoa não reconhecida em área restrita: alerta.

### `app/alert_service.py`

Responsável por exibir mensagens de alerta na tela.

### `app/logger.py`

Responsável por salvar registros do sistema, como horário, pessoa detectada, status e nível de atenção.

### `data/known_faces/`

Pasta onde serão colocadas as fotos das pessoas autorizadas.

Exemplo:

```bash
data/known_faces/integrante_1.jpg
data/known_faces/integrante_2.jpg
```

O nome do arquivo será usado como identificação da pessoa.

### `data/unknown_faces/`

Pasta onde serão salvas imagens de pessoas não reconhecidas.


---

## 8. Fluxo de funcionamento do sistema

```text
1. O sistema é iniciado.
2. A webcam é aberta.
3. O sistema captura imagens em tempo real.
4. A imagem é enviada para o serviço de reconhecimento facial.
5. O sistema compara o rosto com as imagens cadastradas.
6. Se reconhecer a pessoa, exibe status de pessoa autorizada.
7. Se não reconhecer, exibe alerta de atenção.
8. O sistema salva uma imagem como evidência.
9. O resultado aparece na tela para demonstração.
```

Fluxo técnico:

```text
Webcam
  ↓
Camera Service
  ↓
Face Recognition Service
  ↓
Risk Analyzer
  ↓
Alert Service
  ↓
Logger / Evidências
```

---

## 9. Regras de classificação do MVP

O MVP terá uma regra simples de classificação.

| Situação | Status | Nível de atenção |
|---|---|---|
| Pessoa cadastrada reconhecida | Autorizada | Baixo |
| Pessoa não cadastrada | Não reconhecida | Atenção |
| Pessoa não cadastrada em área sensível | Não reconhecida | Alerta |

Para a apresentação, a área monitorada pode ser definida como:

> Entrada principal da residência.

---

## 10. Requisitos funcionais

Os requisitos funcionais são aquilo que o sistema deve fazer.

### RF01 — Abrir webcam

O sistema deve acessar a câmera do computador e exibir o vídeo em tempo real.

### RF02 — Cadastrar pessoas por imagem

O sistema deve considerar como pessoas autorizadas aquelas que possuem imagem cadastrada na pasta `data/known_faces`.

### RF03 — Reconhecer pessoa cadastrada

O sistema deve comparar a imagem capturada pela webcam com a base de rostos cadastrados.

### RF04 — Identificar pessoa não cadastrada

Quando a pessoa não estiver na base de dados local, o sistema deve mostrar que ela não foi reconhecida.

### RF05 — Classificar nível de atenção

O sistema deve informar o nível de atenção com base no resultado do reconhecimento.

### RF06 — Exibir alerta visual

O sistema deve mostrar na tela uma mensagem indicando se a pessoa foi reconhecida ou não.

### RF07 — Salvar evidência

Quando uma pessoa não cadastrada for detectada, o sistema deve salvar uma imagem na pasta `data/unknown_faces` ou `evidencias/prints`.

---

## 11. Requisitos não funcionais

### RNF01 — Simplicidade

O sistema deve ser simples o suficiente para rodar em um computador comum.

### RNF02 — Organização

O código deve ser organizado em módulos separados por responsabilidade.

### RNF03 — Demonstração

O sistema deve permitir gravação de vídeo demonstrando o funcionamento.

### RNF04 — Segurança conceitual

O sistema não deve rotular uma pessoa. Ele apenas classifica pessoas como cadastradas ou não cadastradas.

### RNF05 — Uso local

O MVP deve funcionar localmente, sem depender de servidor externo.

---

## 12. Instalação do projeto

### Passo 1 — Clonar o repositório

```bash
git clone URL_DO_REPOSITORIO
cd face-security-mvp
```

### Passo 2 — Criar ambiente virtual

No Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

No Linux ou macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Passo 3 — Instalar dependências

```bash
pip install -r requirements.txt
```


## 13. Dependências do projeto

Arquivo `requirements.txt`:

```txt
opencv-python
deepface
numpy
pandas
python-dotenv
```

---

## 16. O que precisamos fazer

### 1 — Código base

- [ ] Criar repositório no GitHub.
- [ ] Criar estrutura de pastas.
- [ ] Criar arquivo `main.py`.
- [ ] Criar arquivo `requirements.txt`.
- [ ] Garantir que o projeto rode localmente.
- [ ] Escrever instruções básicas no `README.md`.

### 2 — Reconhecimento facial

- [ ] Instalar e testar DeepFace.
- [ ] Criar pasta `data/known_faces`.
- [ ] Adicionar fotos dos integrantes cadastrados.
- [ ] Testar reconhecimento com uma pessoa cadastrada.
- [ ] Testar pessoa não cadastrada.
- [ ] Registrar prints dos testes.

### 3 — Alertas e evidências

- [ ] Criar regra de risco simples.
- [ ] Exibir mensagem de baixo risco para pessoa cadastrada.
- [ ] Exibir mensagem de atenção para pessoa não cadastrada.
- [ ] Salvar imagem de pessoa não reconhecida.
- [ ] Organizar prints na pasta `evidencias/prints`.
- [ ] Criar pequeno relatório de testes.


---

## 17. Evidências que devem ser guardadas

Para comprovar que o grupo fez o projeto, guardar:

- prints do código no VS Code;
- print da estrutura de pastas;
- print da webcam funcionando;
- print de pessoa cadastrada reconhecida;
- print de pessoa não cadastrada;
- print da pasta com evidências salvas;
- vídeo de funcionamento;
- link do GitHub;
- link do Drive;
- slides da apresentação;
- documentação em Markdown.

Será colocado tudo no google drive.
---

## 23. Tarefas imediatas para começar

1. Testar abertura da webcam.
2. Testar reconhecimento de pessoa cadastrada.
3. Testar pessoa não cadastrada.
4. Salvar prints de evidência.
5. Gravar vídeo de funcionamento.
6. Criar slides.
7. Subir documentação no Drive.
