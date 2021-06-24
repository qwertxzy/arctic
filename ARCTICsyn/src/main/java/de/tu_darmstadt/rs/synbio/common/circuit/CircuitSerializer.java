package de.tu_darmstadt.rs.synbio.common.circuit;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;
import org.jgrapht.io.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;

public class CircuitSerializer extends StdSerializer<Circuit> {

    private static final Logger logger = LoggerFactory.getLogger(CircuitSerializer.class);

    public CircuitSerializer(Class<Circuit> t) {
        super(t);
    }

    @Override
    public void serialize(Circuit circuit, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {

        GateIDProvider gateIDProvider = new GateIDProvider();
        ComponentNameProvider<Wire> wireIDProvider = new IntegerComponentNameProvider<>();

        GateAttributeProvider gateAttributeProvider = new GateAttributeProvider();
        WireAttributeProvider wireAttributeProvider = new WireAttributeProvider();

        JSONExporter<Gate, Wire> exporter = new JSONExporter<>(gateIDProvider, gateAttributeProvider, wireIDProvider, wireAttributeProvider);
        StringWriter writer = new StringWriter();
        try {
            exporter.exportGraph(circuit, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }

        jsonGenerator.writeRawValue(writer.toString());
    }

    static class GateIDProvider implements ComponentNameProvider<Gate> {

        @Override
        public String getName(Gate gate) {
            return gate.getIdentifier();
        }
    }

    static class GateAttributeProvider implements ComponentAttributeProvider<Gate> {

        @Override
        public Map<String, Attribute> getComponentAttributes(Gate gate) {

            HashMap<String, Attribute> map = new HashMap<>();

            String primitiveIdentifier = gate instanceof LogicGate ? ((LogicGate) gate).getLogicType().name() : "";

            map.put("primitiveIdentifier", new DefaultAttribute<>(primitiveIdentifier, AttributeType.STRING));
            map.put("expression", new DefaultAttribute<>(gate.getExpression(), AttributeType.UNKNOWN));
            map.put("type", new DefaultAttribute<>(gate.getType(), AttributeType.UNKNOWN));

            return map;
        }
    }

    static class WireAttributeProvider implements ComponentAttributeProvider<Wire> {

        @Override
        public Map<String, Attribute> getComponentAttributes(Wire wire) {

            HashMap<String, Attribute> map = new HashMap<>();
            map.put("variable", new DefaultAttribute<>(wire.getVariable(), AttributeType.UNKNOWN));
            return map;
        }
    }
}
